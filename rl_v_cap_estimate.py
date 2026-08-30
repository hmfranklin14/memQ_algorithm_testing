"""
rl_v_cap_estimate.py

RLVCapEstimate(Policy) -- wraps the "hint" approach: v_target(t) =
target_input(t) + v_cap_estimate(t) governs the whole episode, except two
short windows immediately after each pulse start/end where an eligible
channel may learn a bounded +/- correction. See the original
v_cap_estimate_env.py's module docstring for the full design rationale.

Owns its own SELF-CONTAINED physics simulation (via rl_common's shared
_physics_step/_smooth_signal_time_major_jit kernels), independent of
physics.py's ModulatorArray -- this is unchanged from how it was trained
as a standalone script, so the already-trained checkpoint stays valid.
Only Policy.evaluate()/plot() (electro-optic math only) are shared with
the other three approaches.

Bump POLICY_VERSION any time observation/action space shape, reward
formula, or network architecture changes -- invalidates the evaluate.py
cache, forcing a retrain.
"""

POLICY_VERSION = "v1"

import os
import shutil
import tempfile

import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from policy import Policy, EnvironmentalConditions
from rl_common import (_physics_step, _smooth_signal_time_major_jit,
                        TrainingMetricsCallback, ProgressBarCallback, BestMSECallback)


class _ModulatorGymEnv(gym.Env):
    metadata = {"render_modes": None}

    def __init__(self, rf_amp, dc_bias, v_pi, null_pt, p_in, er, t_res, mod_res, cap_t, cap_p,
                 mod_cap, ind, ind_p, num_modulators, num_pulses, pulse_width, inter_pulse_gap,
                 pre_pad, post_pad, num_points, rand_power, rand_target, rise_time_ns,
                 edge_window_ns, edge_delta_range_active, edge_delta_range_crosstalk,
                 action_min_v, action_max_v):
        super().__init__()
        self.PER_CHANNEL_OBS_DIM = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_modulators, self.PER_CHANNEL_OBS_DIM), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(num_modulators, dtype=np.float32),
            high=np.ones(num_modulators, dtype=np.float32),
            shape=(num_modulators,), dtype=np.float32)

        self.dc_bias, self.rf_amp, self.v_pi, self.null_pt = dc_bias, rf_amp, v_pi, null_pt
        self.p_in, self.er, self.t_res, self.mod_res = p_in, er, t_res, mod_res
        self.cap_t, self.cap_p, self.mod_cap = cap_t, cap_p, mod_cap
        self.ind, self.ind_p = ind, ind_p
        self.num_modulators, self.num_pulses = num_modulators, num_pulses
        self.pulse_width, self.inter_pulse_gap = pulse_width, inter_pulse_gap
        self.pre_pad, self.post_pad = pre_pad, post_pad
        self.rand_power, self.rand_target, self.rise_time_ns = rand_power, rand_target, rise_time_ns
        self.num_points = num_points
        self.edge_window_ns = edge_window_ns
        self.edge_delta_range_active = edge_delta_range_active
        self.edge_delta_range_crosstalk = edge_delta_range_crosstalk
        self.action_min_v, self.action_max_v = action_min_v, action_max_v

        self.total_time = (pre_pad + num_pulses * pulse_width
                            + (num_pulses - 1) * inter_pulse_gap + post_pad)
        self._times = np.linspace(0, self.total_time, num_points)
        self.h = self._times[1] - self._times[0]

    def _smooth_signal(self, x, rise):
        if rise <= 0:
            return np.array(x, dtype=np.float64, copy=True)
        tau = rise / 2.197
        a = np.exp(-self.h / tau)
        return _smooth_signal_time_major_jit(np.ascontiguousarray(x, dtype=np.float64), a)

    def _calc_power(self, t, v_load):
        pmins = self.p_in / 10**(self._er_vals[t]/10)
        return pmins + (self.p_in - pmins) * np.sin((v_load + self.dc_bias - self.null_pt)
                                                    * np.pi / (2 * self._vpi_vals[t])) ** 2

    def _calc_infidelity(self):
        all_powers = np.array([self._calc_power(s, self._v_mod_hist[s]) for s in range(self.num_points)])
        area_ratio = np.trapezoid(np.sqrt(all_powers), axis=0) / np.trapezoid(np.sqrt(self._target_power), axis=0)
        return np.sin(np.pi*(area_ratio-1)/2)**2

    def _v_target(self):
        s = self.current_step
        return self._target_input[s] + self._v_cap_estimate

    def _get_obs(self):
        s = self.current_step
        elapsed_time = np.full(self.num_modulators, self._times[s] / self.total_time, dtype=np.float32)
        target_power = (self._target_power[s] / self.p_in).astype(np.float32)
        v_target = self._v_target()
        self_active = self._self_active_flag
        adjacent_active_count = self._adjacent_active_count

        rise_pos = self._rise_window_pos[s]
        fall_pos = self._fall_window_pos[s]
        is_rise = 1.0 if rise_pos >= 0 else 0.0
        is_fall = 1.0 if fall_pos >= 0 else 0.0
        window_progress = 0.0
        if rise_pos >= 0:
            window_progress = rise_pos / max(self._window_steps - 1, 1)
        elif fall_pos >= 0:
            window_progress = fall_pos / max(self._window_steps - 1, 1)

        is_rise_window = np.full(self.num_modulators, is_rise, dtype=np.float32)
        is_fall_window = np.full(self.num_modulators, is_fall, dtype=np.float32)
        window_progress_col = np.full(self.num_modulators, window_progress, dtype=np.float32)

        return np.column_stack([elapsed_time, target_power, v_target,
                                 self_active, adjacent_active_count,
                                 is_rise_window, is_fall_window, window_progress_col]).astype(np.float32)

    def _get_info(self):
        n = self.current_step + 1
        mse = self._sq_err_sum / (n * self.num_modulators)
        return {"Current MSE": float(mse)}

    def reset(self, seed=None, options=None, keep_target=False):
        super().reset(seed=seed)
        if not keep_target:
            if self.rand_target:
                self.target_vec = self.np_random.integers(0, 2, self.num_modulators)
                if not self.target_vec.any():
                    force_idx = self.np_random.integers(0, self.num_modulators)
                    self.target_vec[force_idx] = 1
            else:
                self.target_vec = np.ones(self.num_modulators)
        self.p_min = self.p_in / 10**(self.er/10)

        self.current_step = 0
        self._sq_err_sum = 0

        self._self_active_flag = self.target_vec.astype(np.float32)
        adjacent_active_count = np.zeros(self.num_modulators, dtype=np.float32)
        for m in range(self.num_modulators):
            count = 0.0
            if m - 1 >= 0:
                count += self.target_vec[m - 1]
            if m + 1 < self.num_modulators:
                count += self.target_vec[m + 1]
            adjacent_active_count[m] = count
        self._adjacent_active_count = adjacent_active_count

        self._eligible_mask = (self._self_active_flag > 0) | (adjacent_active_count > 0)
        self._edge_delta_scale = np.where(self._self_active_flag > 0,
                                           self.edge_delta_range_active,
                                           self.edge_delta_range_crosstalk).astype(np.float64)

        starts = self.pre_pad + np.arange(self.num_pulses) * (self.pulse_width + self.inter_pulse_gap)
        ends = starts + self.pulse_width
        in_pulse = ((self._times[:, None] >= starts) & (self._times[:, None] < ends)).any(axis=1)
        self.active = in_pulse[:, None] & (self.target_vec[None, :] == 1)

        self._window_steps = max(int(round(self.edge_window_ns / self.h)), 1)
        self._rise_window_pos = np.full(self.num_points, -1, dtype=np.int64)
        self._fall_window_pos = np.full(self.num_points, -1, dtype=np.int64)
        for i in range(self.num_pulses):
            p_start_idx = int(np.searchsorted(self._times, starts[i]))
            p_end_idx = int(np.searchsorted(self._times, ends[i]))
            rise_end = min(p_start_idx + self._window_steps, self.num_points)
            if p_start_idx < rise_end:
                self._rise_window_pos[p_start_idx:rise_end] = np.arange(rise_end - p_start_idx)
            fall_end = min(p_end_idx + self._window_steps, self.num_points)
            if p_end_idx < fall_end:
                self._fall_window_pos[p_end_idx:fall_end] = np.arange(fall_end - p_end_idx)

        self.rc = self.t_res * self.cap_t
        self._v_src = np.zeros((self.num_points, self.num_modulators), dtype=np.float64)

        target_power_unsmoothed = np.where(self.active, self.p_in, self.p_min).astype(np.float64)
        self._target_power_unsmoothed = target_power_unsmoothed  # exposed for Policy.solve() contract
        target_input_unsmoothed = (self.null_pt - self.dc_bias + (2 * self.v_pi / np.pi)
                                    * np.arcsin(np.sqrt((target_power_unsmoothed - self.p_min)
                                                         / (self.p_in - self.p_min))))
        self._target_input = self._smooth_signal(target_input_unsmoothed, self.rise_time_ns)
        self._target_power = (self.p_min + (self.p_in - self.p_min)
                               * np.sin((self._target_input + self.dc_bias - self.null_pt)
                                        * np.pi / (2 * self.v_pi)) ** 2)

        self._vpi_vals = np.ones_like(self._v_src, dtype=np.float64) * self.v_pi
        self._er_vals = np.ones_like(self._v_src, dtype=np.float64) * self.er
        self._v_cap = np.zeros(self.num_modulators, dtype=np.float64)
        self._v_mod_hist = np.zeros_like(self._v_src, dtype=np.float64)
        self._v_cap_estimate = np.zeros(self.num_modulators, dtype=np.float64)

        n = self.num_modulators
        if self.rand_power:
            self._vpi_vals = np.clip(
                self.np_random.normal(self.v_pi, self.v_pi/20, size=(self.num_points, n)),
                0.85*self.v_pi, 1.15*self.v_pi).astype(np.float64)
            self._er_vals = np.clip(
                self.np_random.normal(self.er, self.er/20, size=(self.num_points, n)),
                0.85*self.er, 1.15*self.er).astype(np.float64)
        else:
            self._vpi_vals = np.full((self.num_points, n), self.v_pi, dtype=np.float64)
            self._er_vals = np.full((self.num_points, n), self.er, dtype=np.float64)

        self.g_mat = np.diag([1 / self.t_res] * n)
        self.m_mat = np.zeros((n, n))
        self.m_mat[-1][-1] = self.cap_t + self.cap_p
        for i in range(n - 1):
            self.m_mat[i][i] = self.cap_t + 2 * self.cap_p
            self.m_mat[i + 1][i] = -self.cap_p
            self.m_mat[i][i + 1] = -self.cap_p
        self.m_mat[0][0] -= self.cap_p
        self.a_inv = np.linalg.inv(self.m_mat + self.h * self.g_mat)

        self.l_mat = np.zeros((n, n))
        np.fill_diagonal(self.l_mat, self.ind)
        for i in range(n - 1):
            self.l_mat[i][i + 1] += self.ind_p
            self.l_mat[i + 1][i] += self.ind_p

        self.r2_mat = np.diag([self.mod_res] * n)
        self.c_mod_mat = np.diag([self.mod_cap] * n)

        big2 = np.zeros((2 * n, 2 * n))
        big2[:n, :n] = self.c_mod_mat
        big2[:n, n:] = -self.h * np.eye(n)
        big2[n:, :n] = self.h * np.eye(n)
        big2[n:, n:] = self.l_mat + self.h * self.r2_mat
        self.big2_inv = np.linalg.inv(big2)

        self._v_mod = np.zeros(n)
        self._i_ind = np.zeros(n)

        return self._get_obs(), self._get_info()

    def step(self, action):
        s = self.current_step
        raw_action = np.clip(action, -1.0, 1.0).astype(np.float64)

        v_target = self._v_target()
        in_window = (self._rise_window_pos[s] >= 0) | (self._fall_window_pos[s] >= 0)
        apply_correction = self._eligible_mask & in_window
        delta = raw_action * self._edge_delta_scale
        v = np.where(apply_correction, v_target + delta, v_target)
        v = np.clip(v, self.action_min_v, self.action_max_v)

        self._v_src[s] = v
        self._v_cap_estimate = self._v_cap_estimate + (self.h / self.rc) * (v - self._v_cap_estimate)

        (self._v_cap, self._v_mod, self._i_ind, power, reward) = _physics_step(
            v, self._v_cap, self._v_mod, self._i_ind,
            self.a_inv, self.m_mat, self.g_mat, self.h, self.rc, self.cap_p == 0,
            self.l_mat, self.c_mod_mat, self.big2_inv,
            self.p_in, self.null_pt, self.dc_bias,
            self._vpi_vals[s], self._er_vals[s], self._target_power[s],
        )
        self._v_mod_hist[s] = self._v_mod
        self._sq_err_sum += reward

        self.current_step += 1
        terminated = self.current_step >= self.num_points

        obs = self._get_obs() if not terminated else np.zeros(
            (self.num_modulators, self.PER_CHANNEL_OBS_DIM), dtype=np.float32)
        return obs, reward, terminated, False, self._get_info()

    def close(self):
        pass


class RLVCapEstimate(Policy):
    def __init__(self,
                 conditions: EnvironmentalConditions,
                 edge_window_ns: float = 0.3,
                 edge_delta_range_active: float = 0.3,
                 edge_delta_range_crosstalk: float = 0.01,
                 action_min_v: float = 0.0,
                 action_max_v: float = 1.6,
                 rand_power: bool = False,
                 total_timesteps: int = 4_000_000,
                 learning_rate: float = 0.0001319958418905529,
                 n_steps: int = 5000,
                 batch_size: int = 500,
                 n_epochs: int = 6,
                 ent_coef: float = 0.026812687114767707,
                 clip_range: float = 0.30336866001652685,
                 lstm_hidden_size: int = 64,
                 n_lstm_layers: int = 1,
                 net_arch: dict | None = None,
                 eval_freq: int = 100,
                 n_eval_episodes: int = 10):
        super().__init__(conditions)
        self.edge_window_ns = edge_window_ns
        self.edge_delta_range_active = edge_delta_range_active
        self.edge_delta_range_crosstalk = edge_delta_range_crosstalk
        self.action_min_v = action_min_v
        self.action_max_v = action_max_v
        self.rand_power = rand_power
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.net_arch = net_arch if net_arch is not None else dict(pi=[256], vf=[256])
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.model: RecurrentPPO | None = None
        self.vec_normalize: VecNormalize | None = None

    def _build_env(self) -> _ModulatorGymEnv:
        c = self.conditions
        return _ModulatorGymEnv(
            rf_amp=c.rf_amp, dc_bias=c.dc_bias, v_pi=c.v_pi, null_pt=c.null_pt, p_in=c.p_in, er=c.er,
            t_res=c.t_res, mod_res=c.mod_res, cap_t=c.cap_t, cap_p=c.cap_p, mod_cap=c.mod_cap,
            ind=c.ind, ind_p=c.ind_p, num_modulators=c.num_modulators, num_pulses=c.num_pulses,
            pulse_width=c.pulse_width, inter_pulse_gap=c.inter_pulse_gap, pre_pad=c.pre_pad,
            post_pad=c.post_pad, num_points=c.num_points, rand_power=self.rand_power,
            rand_target=True, rise_time_ns=c.rise_time_ns,
            edge_window_ns=self.edge_window_ns, edge_delta_range_active=self.edge_delta_range_active,
            edge_delta_range_crosstalk=self.edge_delta_range_crosstalk,
            action_min_v=self.action_min_v, action_max_v=self.action_max_v,
        )

    def _build_vec_env(self, training: bool) -> VecNormalize:
        env = self._build_env()
        return VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=training)

    def train(self, **kwargs) -> None:
        train_vec_env = self._build_vec_env(training=True)
        eval_env = self._build_env()
        eval_vec_env = self._build_vec_env(training=False)

        model = RecurrentPPO(
            "MlpLstmPolicy", train_vec_env, verbose=0,
            policy_kwargs=dict(lstm_hidden_size=self.lstm_hidden_size,
                                n_lstm_layers=self.n_lstm_layers, net_arch=self.net_arch),
            learning_rate=self.learning_rate, n_steps=self.n_steps, batch_size=self.batch_size,
            n_epochs=self.n_epochs, ent_coef=self.ent_coef, clip_range=self.clip_range,
        )

        best_ckpt_dir = tempfile.mkdtemp(prefix="rl_v_cap_estimate_best_")
        try:
            best_cb = BestMSECallback(eval_vec_env, eval_env, best_ckpt_dir,
                                       eval_freq=self.eval_freq, n_eval_episodes=self.n_eval_episodes)
            callbacks = [TrainingMetricsCallback(), ProgressBarCallback(self.total_timesteps), best_cb]
            model.learn(total_timesteps=self.total_timesteps, callback=callbacks)

            if best_cb.found_best:
                print(f"Reloading best checkpoint (MSE={best_cb.best_mse:.6f}).")
                reload_env = self._build_env()
                reloaded_vec = VecNormalize.load(
                    os.path.join(best_ckpt_dir, "vecnormalize.pkl"), DummyVecEnv([lambda: reload_env]))
                reloaded_vec.training = False
                reloaded_vec.norm_reward = False
                self.model = RecurrentPPO.load(os.path.join(best_ckpt_dir, "model"), env=reloaded_vec)
                self.vec_normalize = reloaded_vec
            else:
                print("No evaluation episode ever completed -- keeping the final training state.")
                self.model = model
                self.vec_normalize = train_vec_env
        finally:
            shutil.rmtree(best_ckpt_dir, ignore_errors=True)

    def save(self, path: str) -> None:
        if self.model is None or self.vec_normalize is None:
            raise RuntimeError("RLVCapEstimate.save() called before train(); nothing to save.")
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model"))
        self.vec_normalize.save(os.path.join(path, "vecnormalize.pkl"))

    def load(self, path: str) -> None:
        env = self._build_env()
        vec_normalize = VecNormalize.load(os.path.join(path, "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        self.model = RecurrentPPO.load(os.path.join(path, "model"), env=vec_normalize)
        self.vec_normalize = vec_normalize

    def solve(self, active_channels: list[int] | None = None):
        if self.model is None or self.vec_normalize is None:
            raise RuntimeError("RLVCapEstimate.solve() called before train()/load(); no model available.")

        env = self._build_env()
        n = self.conditions.num_modulators
        target_vec = np.ones(n, dtype=int) if active_channels is None else np.zeros(n, dtype=int)
        if active_channels is not None:
            target_vec[np.asarray(active_channels, dtype=int)] = 1
        env.target_vec = target_vec
        obs, _ = env.reset(keep_target=True)

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        while True:
            norm_obs = self.vec_normalize.normalize_obs(obs[None, ...])
            action, lstm_states = self.model.predict(
                norm_obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            episode_starts = np.zeros((1,), dtype=bool)
            if terminated or truncated:
                break

        # env's own arrays are TIME-MAJOR (T, n); Policy contract is
        # channel-major (n, T) -- transpose at this boundary.
        return env._v_mod_hist.T, env._v_src.T, env.active.T, env._target_power_unsmoothed.T