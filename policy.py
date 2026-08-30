"""
policy.py

Shared base class for both approaches. Owns everything that must be
IDENTICAL between classical control and RL: environment/spec construction,
the rise-time smoothing applied to the target trajectory, scoring
(MSE + infidelity, split by active/inactive channels), and the unified
two-line (target vs. actual) plot. Subclasses (ClassicalControl, RL)
implement only what's genuinely different: how they train (or don't) and
how they solve. A classical-only raw-baseline comparison plot lives in
classical_control.py, not here, since RL has no equivalent baseline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from numba import njit

from physics import ModulatorArray

BRAND_PURPLE = "#73368b"
BRAND_BLACK = "#000000"

@njit(cache=True)
def _smooth_signal_uniform_jit(x, a):
    """
    Per-step recursion y[k+1] = a*y[k] + (1-a)*x[k+1] -- the exact solution
    of a first-order lag on a uniform time grid, where a = exp(-dt/tau).
    """
    n, T = x.shape
    y = np.empty_like(x)
    y[:, 0] = x[:, 0]
    for k in range(T - 1):
        y[:, k + 1] = a * y[:, k] + (1 - a) * x[:, k + 1]
    return y


@dataclass
class EnvironmentalConditions:
    """
    Everything needed to build a ModulatorArray + target trajectory,
    common to both approaches. RL-specific voltage-range parameters
    (on/off/pre/never-fire) do NOT live here -- see rl.py.
    """
    num_modulators: int = 3
    h: float = 1.0                     # timestep, ns (native ModulatorArray unit)
    num_pulses: int = 1
    pulse_width: float = 150.0         # ns
    inter_pulse_gap: float = 1000.0    # ns
    pre_pad: float = 100.0             # ns
    post_pad: float = 100.0            # ns
    num_points: int = 350
    rise_time_ns: float = 10.0         # 10 ns default per engineering guidance
    # Circuit / electro-optic parameters, passed straight through to ModulatorArray:
    transition_window_ns: float = 10.0
    rf_amp: float = 1.3
    dc_bias: float = 0.8
    v_pi: float = 1.3
    null_pt: float = 0.8
    p_in: float = 1000
    er: float = 29.6
    t_res: float = 50
    mod_res: float = 50
    cap_t: float = 50
    cap_p: float = 0.0001
    mod_cap: float = 0.0003
    ind: float = 2.7
    ind_p: float = 1
    skip_cap: bool = False


class Policy(ABC):
    """
    Base class inherited by ClassicalControl and RL.

    Lifecycle:
        policy = SomeSubclass(conditions, **subclass_specific_kwargs)
        policy.train()                                  # no-op for classical, RecurrentPPO.learn for RL
        v_mod_hist, v_src, active, target_powers = policy.solve(active_channels=[0, 2])
        results = policy.evaluate(v_mod_hist, active, target_powers)
        policy.plot(v_src, v_mod_hist, target_powers)

    solve() returns active/target_powers alongside v_mod_hist/v_src because
    it necessarily builds both internally to do its own work (classical
    needs target_input to invert against; RL needs target_powers for
    reward/observations during rollout) -- returning them avoids evaluate.py
    recomputing the same active mask and smoothed target a second time.
    """

    def __init__(self, conditions: EnvironmentalConditions):
        self.conditions = conditions
        self.mod_array = ModulatorArray(
            num_modulators=conditions.num_modulators,
            h=conditions.h,
            rf_amp=conditions.rf_amp,
            dc_bias=conditions.dc_bias,
            v_pi=conditions.v_pi,
            null_pt=conditions.null_pt,
            p_in=conditions.p_in,
            er=conditions.er,
            t_res=conditions.t_res,
            mod_res=conditions.mod_res,
            cap_t=conditions.cap_t,
            cap_p=conditions.cap_p,
            mod_cap=conditions.mod_cap,
            ind=conditions.ind,
            ind_p=conditions.ind_p,
            skip_cap=conditions.skip_cap,
        )
        self.times = self._build_time_grid()

    # ---- shared, concrete: environment/spec construction ----

    def _build_time_grid(self) -> npt.NDArray[np.float64]:
        """
        Builds the (num_points,) time array in ns, spanning
        pre_pad + num_pulses*pulse_width + (num_pulses-1)*inter_pulse_gap + post_pad.
        Also sets self.total_time for reference.
        """
        c = self.conditions
        self.total_time = (c.pre_pad
                            + c.num_pulses * c.pulse_width
                            + (c.num_pulses - 1) * c.inter_pulse_gap
                            + c.post_pad)
        return np.linspace(0, self.total_time, c.num_points)

    def build_active_mask(self, active_channels: list[int] | None) -> npt.NDArray[np.bool_]:
        """
        Build the (n, T) boolean active mask for the given set of firing
        channels, using this Policy's pre_pad/pulse_width/inter_pulse_gap/
        num_pulses timing. active_channels=None means every channel fires.

        This is the fixed-pattern case used by both ClassicalControl.solve()
        and RL.solve() at evaluation time. RL's training-time random pattern
        selection is handled separately inside _ModulatorGymEnv.reset(),
        which calls this same method once a target_vec has been drawn.
        """
        c = self.conditions
        n = c.num_modulators
        T = c.num_points

        if active_channels is None:
            fires = np.ones(n, dtype=bool)
        else:
            fires = np.zeros(n, dtype=bool)
            fires[np.asarray(active_channels, dtype=int)] = True

        starts = c.pre_pad + np.arange(c.num_pulses) * (c.pulse_width + c.inter_pulse_gap)
        ends = starts + c.pulse_width
        in_pulse = ((self.times[:, None] >= starts) & (self.times[:, None] < ends)).any(axis=1)  # (T,)

        active = in_pulse[:, None] & fires[None, :]  # (T, n)
        return active.T  # (n, T) -- channel-major, matching ModulatorArray/FF_Decoupler convention

    def build_targets(self, active: npt.NDArray[np.bool_]):
        target_powers_unsmoothed = np.where(active, self.mod_array.p_in, self.mod_array.p_min)
        target_input_unsmoothed = self.mod_array.calc_power_inv(target_powers_unsmoothed)
        target_input_smoothed = self._smooth_signal(target_input_unsmoothed, self.conditions.rise_time_ns)
        target_powers_smoothed = self.mod_array.calc_power(target_input_smoothed)
        return target_powers_smoothed, target_input_smoothed, target_powers_unsmoothed, target_input_unsmoothed

    def _smooth_signal(self, x: npt.NDArray[np.float64], rise: float) -> npt.NDArray[np.float64]:
        """
        First-order-lag smoothing, tau = rise / 2.197 (10%-90% rise-time
        convention). x is channel-major, shape (n, T). Operates in ns,
        matching self.times -- ClassicalControl converts to/from SI
        seconds at its own boundary, not here.
        """
        if rise <= 0:
            return np.array(x, dtype=np.float64, copy=True)
        tau = rise / 2.197
        dt = np.diff(self.times)

        if len(dt) > 0 and np.allclose(dt, dt[0], atol=0):
            a = np.exp(-dt[0] / tau)
            return _smooth_signal_uniform_jit(np.ascontiguousarray(x, dtype=np.float64), a)

        y = np.zeros_like(x, dtype=np.float64)
        y[:, 0] = x[:, 0]
        for k in range(len(self.times) - 1):
            a = np.exp(-dt[k] / tau)
            y[:, k + 1] = a * y[:, k] + (1 - a) * x[:, k + 1]
        return y

    # ---- shared, concrete: scoring ----
    def evaluate(self, v_mod_hist: npt.NDArray[np.float64], active: npt.NDArray[np.bool_],
                 target_powers: npt.NDArray[np.float64]) -> dict:
        """
        Computes, identically for both approaches:
            - per-channel MSE, overall MSE (full episode)
            - per-channel masked MSE, overall mean masked MSE -- excludes,
              for each channel, the fixed transition_window_ns immediately
              following every rising transition (pulse start) and every
              falling transition (pulse end) in that channel's active mask.
              A channel that never fires has no transitions, so its masked
              MSE equals its full-series MSE.
            - per-channel transition MSE, overall mean transition MSE --
              the complement of masked MSE: computed ONLY over those
              transition_window_ns windows. A channel that never fires has
              no such window, so its transition MSE is NaN; the overall
              mean ignores NaNs (i.e. only averages over channels that
              actually fired).
            - per-channel infidelity (against the SMOOTHED target_powers)
            - mean infidelity overall (no active/inactive split)

        v_mod_hist, active, target_powers all channel-major, shape (n, T).
        """
        power_hist = self.mod_array.calc_power(v_mod_hist)
        sq_err = (target_powers - power_hist) ** 2

        mse_per_channel = np.mean(sq_err, axis=1)
        mse_overall = float(np.mean(mse_per_channel))

        mse_masked_per_channel, mse_transition_per_channel = self._compute_masked_and_transition_mse(active, sq_err)
        mse_masked_overall = float(np.mean(mse_masked_per_channel))
        mse_transition_overall = float(np.nanmean(mse_transition_per_channel))

        infidelity = self.mod_array.calc_infidelity(v_mod_hist, target_powers)
        mean_infidelity_overall = float(np.mean(infidelity))

        return {
            "mse_per_channel": mse_per_channel,
            "mse_overall": mse_overall,
            "mse_masked_per_channel": mse_masked_per_channel,
            "mse_masked_overall": mse_masked_overall,
            "mse_transition_per_channel": mse_transition_per_channel,
            "mse_transition_overall": mse_transition_overall,
            "infidelity_per_channel": infidelity,
            "mean_infidelity_overall": mean_infidelity_overall,
        }

    def _compute_masked_and_transition_mse(self, active: npt.NDArray[np.bool_],
                                            sq_err: npt.NDArray[np.float64]):
        """
        Per-channel masked MSE (excludes fixed transition_window_ns windows
        following every rising/falling edge of that channel's active mask)
        and per-channel transition MSE (computed ONLY over those same
        windows -- the complement).

        A channel with no transitions (never fires) excludes nothing, so
        its masked MSE equals its full MSE; it also has no transition
        window, so its transition MSE is NaN.
        """
        n, T = active.shape
        window_steps = int(round(self.conditions.transition_window_ns / self.conditions.h))

        in_window = np.zeros((n, T), dtype=bool)
        for m in range(n):
            diffs = np.diff(active[m].astype(np.int8))
            transition_idx = np.where(diffs != 0)[0] + 1  # rising (+1) and falling (-1) edges alike
            for idx in transition_idx:
                end = min(idx + window_steps, T)
                in_window[m, idx:end] = True

        mse_masked = np.array([
            np.mean(sq_err[m, ~in_window[m]]) if (~in_window[m]).any() else float(np.mean(sq_err[m]))
            for m in range(n)
        ])
        mse_transition = np.array([
            np.mean(sq_err[m, in_window[m]]) if in_window[m].any() else np.nan
            for m in range(n)
        ])

        return mse_masked, mse_transition

    # ---- shared, concrete: plotting ----

    def plot(self, v_src: npt.NDArray[np.float64], v_mod_hist: npt.NDArray[np.float64],
              target_powers: npt.NDArray[np.float64]):
        """
        Unified THREE-panel plot (input voltage command, optical power vs.
        target, residual), paginated one channel at a time with Prev/Next
        buttons -- used identically by all three RL approaches.
        """
        n = self.conditions.num_modulators
        power_hist = self.mod_array.calc_power(v_mod_hist)
        residual = power_hist - target_powers

        fig, (ax_in, ax_power, ax_res) = plt.subplots(1, 3, figsize=(15, 4.2))
        plt.subplots_adjust(bottom=0.28, top=0.8, wspace=0.4)

        legend_handles = [
            Line2D([0], [0], color=BRAND_BLACK, linestyle="--", linewidth=2, label="target"),
            Line2D([0], [0], color=BRAND_PURPLE, label="actual"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.12))

        state = {"i": 0}

        def draw_channel(i):
            for ax in (ax_in, ax_power, ax_res):
                ax.clear()
                ax.tick_params(axis="y", labelsize=8)

            ax_in.plot(self.times, v_src[i], color=BRAND_PURPLE)
            ax_in.set_title("Input voltage (command)", pad=15)
            ax_in.set_xlabel("time (ns)")
            ax_in.set_ylabel("V")

            ax_power.plot(self.times, target_powers[i], color=BRAND_BLACK, linestyle="--", linewidth=2)
            ax_power.plot(self.times, power_hist[i], color=BRAND_PURPLE)
            ax_power.set_title("Optical power vs. target", pad=15)
            ax_power.set_xlabel("time (ns)")
            ax_power.set_ylabel("nW")

            ax_res.axhline(0, color=BRAND_BLACK, linestyle="--", linewidth=2)
            ax_res.scatter(self.times, residual[i], color=BRAND_PURPLE, alpha=0.5, s=5)
            ax_res.set_title("Optical power residual vs. target", pad=15)
            ax_res.set_xlabel("time (ns)")
            ax_res.set_ylabel("nW")

            fig.suptitle(f"Channel {i + 1} of {n}", y=0.92)
            fig.canvas.draw_idle()

        draw_channel(state["i"])

        ax_prev = fig.add_axes([0.35, 0.03, 0.1, 0.06])
        ax_next = fig.add_axes([0.55, 0.03, 0.1, 0.06])
        btn_prev = Button(ax_prev, "< Prev")
        btn_next = Button(ax_next, "Next >")

        def on_prev(_event):
            state["i"] = (state["i"] - 1) % n
            draw_channel(state["i"])

        def on_next(_event):
            state["i"] = (state["i"] + 1) % n
            draw_channel(state["i"])

        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

        plt.show(block=False)
        return btn_prev, btn_next

    # ---- subclass-specific: must override ----

    @abstractmethod
    def train(self, **kwargs) -> None:
        """No-op for ClassicalControl. Runs RecurrentPPO.learn(...) for RL."""
        ...

    @abstractmethod
    def solve(self, active_channels: list[int] | None = None):
        """
        Returns (v_mod_hist, v_src, active, target_powers):
            v_mod_hist, v_src, active, target_powers -- all shape (n, T),
            channel-major. v_mod_hist/v_src are the modulator voltage
            history and raw input voltage command history resulting from
            this policy's strategy, given a fixed activation pattern
            (active_channels). active/target_powers are exactly what
            build_active_mask()/build_targets() would produce for the same
            active_channels -- returned here rather than recomputed by the
            caller, since solve() already builds both internally.

        Both subclasses accept the same active_channels parameter and
        return the same 4-tuple shape so evaluate.py and plot() never need
        to branch on approach.
        """
        ...