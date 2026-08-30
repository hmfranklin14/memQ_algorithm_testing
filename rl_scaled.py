"""
rl_scaled.py

RLScaled(Policy) -- the multi-branch, phase-dependent action-space design
(pre/on/off/never-fire, each with its own absolute voltage range;
strictly-increasing accumulator during on-pulse), paired with the same
"hammer, not magic wand" structured observation as rl_standard.py. See the
original narrow_env.py's module docstring for full rationale.
"""

POLICY_VERSION = "v1"

import os
import shutil
import tempfile

import numpy as np
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
                 on_v_range, off_v_range, pre_v_range, never_fire_v_range, on_pulse_min_increment):
        super().__init__()
        self.PER_CHANNEL_OBS_DIM = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_modulators, self.PER_CHANNEL_OBS_DIM), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(num_modulators), high=np.ones(num_modulators),
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
        self.on_v_range, self.off_v_range = on_v_range, off_v_range
        self.pre_v_range, self.never_fire_v_range = pre_v_range, never_fire_v_range
        self.on_pulse_min_increment = on_pulse_min_increment

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

    def _get_obs(self):
        s = self.current_step
        elapsed_time = np.full(self.num_modulators, self._times[s], dtype=np.float32)
        target_power = self._target_power[s]
        current_power = self._calc_power(s, self._v_mod) if s > 0 else np.zeros(self.num_modulators)
        current_v_src = self._v_src[s - 1] if s > 0 else np.zeros(self.num_modulators)

        in_pulse = (target_power > self.p_min + (self.p_in - self.p_min) * 0.5).astype(np.float32)
        pulse_phase = np.zeros(self.num_modulators, dtype=np.float32)
        if np.any(in_pulse):
            pulse_phase = in_pulse * (self._pulse_step_count[s] / self._steps_per_pulse)
        has_fired_before = (np.any(self.active[:s, :], axis=0).astype(np.float32)
                            if s > 0 else np.zeros(self.num_modulators, dtype=np.float32))
        post_pulse_off = has_fired_before * (1.0 - in_pulse)

        adjacent_v_src = np.zeros(self.num_modulators, dtype=np.float32)
        adjacent_v_src[1:] += current_v_src[:-1]
        adjacent_v_src[:-1] += current_v_src[1:]

        return np.column_stack([elapsed_time, target_power, current_power, current_v_src,
                                 adjacent_v_src, pulse_phase, in_pulse,
                                 post_pulse_off]).astype(np.float32)

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
        self._never_fires_mask = (self.target_vec == 0)
        self.p_min = self.p_in / 10**(self.er/10)

        self.current_step = 0
        self._sq_err_sum = 0

        starts = self.pre_pad + np.arange(self.num_pulses) * (self.pulse_width + self.inter_pulse_gap)
        ends = starts + self.pulse_width
        in_pulse = ((self._times[:, None] >= starts) & (self._times[:, None] < ends)).any(axis=1)
        self._pulse_step_count = np.zeros(self.num_points, dtype=np.int64)
        for i in range(self.num_pulses):
            p_start_idx = np.searchsorted(self._times, starts[i])
            p_end_idx = np.searchsorted(self._times, ends[i])
            self._pulse_step_count[p_start_idx:p_end_idx] = np.arange(1, p_end_idx - p_start_idx + 1)
        self._steps_per_pulse = max(int(np.ceil(self.pulse_width / self.h)), 1)
        self.active = in_pulse[:, None] & (self.target_vec[None, :] == 1)

        self._global_first_fire_idx = (
            int(np.searchsorted(self._times, starts[0])) if self.num_pulses > 0 else self.num_points
        )

        self._on_pulse_accum = np.zeros(self.num_modulators, dtype=np.float64)

        self.rc = self.t_res * self.cap_t
        self._v_src = np.zeros((self.num_points, self.num_modulators), dtype=np.float64)

        target_power_unsmoothed = np.where(self.active, self.p_in, self.p_min).astype(np.float64)
        self._target_power_unsmoothed = target_power_unsmoothed  # exposed for Policy.solve() contract
        target_input_unsmoothed = (self.null_pt - self.dc_bias + (2 * self.v_pi / np.pi)
                                    * np.arcsin(np.sqrt((target_power_unsmoothed - self.p_min)
                                                         / (self.p_in - self.p_min))))
        target_input_smoothed = self._smooth_signal(target_input_unsmoothed, self.rise_time_ns)
        self._target_power = (self.p_min + (self.p_in - self.p_min)
                               * np.sin((target_input_smoothed + self.dc_bias - self.null_pt)
                                        * np.pi / (2 * self.v_pi)) ** 2)

        self._vpi_vals = np.ones_like(self._v_src, dtype=np.float64) * self.v_pi
        self._er_vals = np.ones_like(self._v_src, dtype=np.float64) * self.er
        self._v_cap = np.zeros(self.num_modulators, dtype=np.float64)
        self._v_mod_hist = np.zeros_like(self._v_src, dtype=np.float64)

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

    def _map_action(self, raw_action, s):
        normalized = (raw_action + 1.0) / 2.0

        if s < self._global_first_fire_idx:
            lo = np.full(self.num_modulators, self.pre_v_range[0])
            hi = np.full(self.num_modulators, self.pre_v_range[1])
            in_pulse_mask = np.zeros(self.num_modulators, dtype=bool)
        else:
            in_pulse_mask = self.active[s]
            on_lo, on_hi = self.on_v_range
            off_lo, off_hi = self.off_v_range
            nf_lo, nf_hi = self.never_fire_v_range

            lo = np.where(self._never_fires_mask, nf_lo, np.where(in_pulse_mask, on_lo, off_lo))
            hi = np.where(self._never_fires_mask, nf_hi, np.where(in_pulse_mask, on_hi, off_hi))

        v_default = lo + normalized * (hi - lo)

        starting_pulse = in_pulse_mask & (self._pulse_step_count[s] == 1)
        self._on_pulse_accum[starting_pulse] = 0.0

        increment = self.on_pulse_min_increment + (1.0 - self.on_pulse_min_increment) * normalized
        self._on_pulse_accum = np.where(in_pulse_mask, self._on_pulse_accum + increment, self._on_pulse_accum)

        on_lo, on_hi = self.on_v_range
        on_pulse_voltage = on_lo + (self._on_pulse_accum / self._steps_per_pulse) * (on_hi - on_lo)

        return np.where(in_pulse_mask, on_pulse_voltage, v_default)

    def step(self, action):
        s = self.current_step
        raw_action = np.clip(action, -1.0, 1.0).astype(np.float64)
        v = self._map_action(raw_action, s)

        self._v_src[s] = v

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


class RLScaled(Policy):
    def __init__(self,
                 conditions: EnvironmentalConditions,
                 on_v_range: tuple[float, float] = (1.3, 1.38),
                 off_v_range: tuple[float, float] = (0.0, 0.08),
                 pre_v_range: tuple[float, float] = (0.0, 0.0),
                 never_fire_v_range: tuple[float, float] = (0.0, 1.0),
                 on_pulse_min_increment: float = 0.01,
                 rand_power: bool = False,
                 total_timesteps: int = 2_000_000,
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
        self.on_v_range = on_v_range
        self.off_v_range = off_v_range
        self.pre_v_range = pre_v_range
        self.never_fire_v_range = never_fire_v_range
        self.on_pulse_min_increment = on_pulse_min_increment
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
            on_v_range=self.on_v_range, off_v_range=self.off_v_range,
            pre_v_range=self.pre_v_range, never_fire_v_range=self.never_fire_v_range,
            on_pulse_min_increment=self.on_pulse_min_increment,
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

        best_ckpt_dir = tempfile.mkdtemp(prefix="rl_scaled_best_")
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
            raise RuntimeError("RLScaled.save() called before train(); nothing to save.")
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
            raise RuntimeError("RLScaled.solve() called before train()/load(); no model available.")

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

        return env._v_mod_hist.T, env._v_src.T, env.active.T, env._target_power_unsmoothed.T