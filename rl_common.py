"""
rl_common.py

Shared infrastructure for all three RL approaches (rl_v_cap_estimate.py,
rl_standard.py, rl_scaled.py): the physics kernel and training callbacks
were byte-for-byte identical across all three original standalone
training scripts, so they live here once instead of being tripled.

NOTE ON PHYSICS: this _physics_step models cap_p as coupling the BIAS-TEE
stage (m_mat), NOT the LC/modulator-capacitance stage the way physics.py's
ModulatorArray does. This is a genuine, pre-existing difference between
how these three RL approaches were trained and how physics.py/
classical_control.py model crosstalk -- not a bug introduced here. It
doesn't affect scoring (Policy.evaluate()/plot() only ever call
mod_array.calc_power()/calc_infidelity(), which depend only on the
electro-optic parameters -- v_pi, null_pt, p_in, er, dc_bias -- never on
t_res/cap_t/cap_p/ind/ind_p/mod_cap), but it means "the same cap_p value"
does NOT mean "the same physical coupling assumption" across approaches.
Worth remembering before treating cap_p as directly comparable across
classical control and these three RL policies.
"""

import os

import numpy as np
from tqdm import tqdm
from numba import njit
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization


@njit(cache=True)
def _physics_step(v_action, v_cap, v_mod, i_ind,
                a_inv, m_mat, g_mat, h, rc, cap_p_is_zero,
                l_mat, c_mod_mat, big2_inv,
                p_in, null_pt, dc_bias, vpi_val, er_val, target_val):
    n = v_action.shape[0]

    if cap_p_is_zero:
        v_cap_new = v_cap + (h / rc) * (v_action - v_cap)
    else:
        v_cap_new = a_inv @ (m_mat @ v_cap + h * (g_mat @ v_action))

    v_load = v_action - v_cap_new

    rhs = np.empty(2 * n, dtype=np.float64)
    rhs[:n] = c_mod_mat @ v_mod
    rhs[n:] = l_mat @ i_ind + h * v_load

    y_new = big2_inv @ rhs
    v_mod_new = y_new[:n]
    i_ind_new = y_new[n:]

    p_min = p_in / 10 ** (er_val / 10)
    power = p_min + (p_in - p_min) * np.sin(
        (v_mod_new + dc_bias - null_pt) * np.pi / (2 * vpi_val)
    ) ** 2

    err = (target_val - power) / p_in
    sq_err = err ** 2

    reward = -np.mean(sq_err)

    return v_cap_new, v_mod_new, i_ind_new, power, reward


@njit(cache=True)
def _smooth_signal_time_major_jit(x, a):
    T, n = x.shape
    y = np.empty_like(x)
    y[0, :] = x[0, :]
    for k in range(T - 1):
        y[k + 1, :] = a * y[k, :] + (1 - a) * x[k + 1, :]
    return y


class TrainingMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]
        if "episode" in info:
            self.episode_rewards.append(info["episode"]["r"])
            self._current_reward = 0.0
        elif done:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


class BestMSECallback(BaseCallback):
    """
    Periodically runs n_eval_episodes full episodes -- each with its own
    freshly-drawn random firing pattern -- and computes the mean squared
    error (target vs. actual power, averaged over ALL channels) for each
    one, then averages across all n_eval_episodes runs. Whenever that
    averaged MSE improves on the best seen so far, saves the current model
    weights + matching VecNormalize stats to best_ckpt_dir.

    No early stopping -- matches the regime that actually produced the
    currently-trained checkpoints (every standalone training run used its
    full total_timesteps; none stopped early).
    """

    def __init__(self, eval_env, base_eval_env, best_ckpt_dir, eval_freq=100, n_eval_episodes=10):
        super().__init__()
        self.eval_env = eval_env
        self.base_eval_env = base_eval_env
        self.best_ckpt_dir = best_ckpt_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mse = float("inf")
        self.found_best = False
        self._episode_count = 0

    def _run_one_eval_episode(self) -> float:
        sync_envs_normalization(self.model.get_env(), self.eval_env)

        obs, info = self.base_eval_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while True:
            norm_obs = self.eval_env.normalize_obs(obs[None, ...])
            action, lstm_states = self.model.predict(
                norm_obs, state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, terminated, truncated, info = self.base_eval_env.step(action[0])
            episode_starts = np.zeros((1,), dtype=bool)
            if terminated or truncated:
                break

        power_after = np.array([self.base_eval_env._calc_power(s, v)
                                for s, v in enumerate(self.base_eval_env._v_mod_hist)])
        mse_per = np.mean((self.base_eval_env._target_power - power_after) ** 2, axis=0)
        return float(np.mean(mse_per))

    def _run_eval_mse(self) -> float:
        per_episode_mse = [self._run_one_eval_episode() for _ in range(self.n_eval_episodes)]
        return float(np.mean(per_episode_mse))

    def _save_best_checkpoint(self) -> None:
        os.makedirs(self.best_ckpt_dir, exist_ok=True)
        self.model.save(os.path.join(self.best_ckpt_dir, "model"))
        self.model.get_env().save(os.path.join(self.best_ckpt_dir, "vecnormalize.pkl"))

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]
        if done:
            self._episode_count += 1
            if self._episode_count % self.eval_freq == 0:
                mse = self._run_eval_mse()
                print(f"\nEval MSE (avg over {self.n_eval_episodes} episodes) at episode "
                      f"{self._episode_count}: {mse:.12f}")
                if mse < self.best_mse:
                    self.best_mse = mse
                    self._save_best_checkpoint()
                    self.found_best = True
                    print(f"New best MSE {mse:.12f} -- checkpoint saved")
        return True  # never stop early