"""
classical_control.py

ClassicalControl(Policy) -- the exact linear-model-inversion approach,
adapted from FF_Decoupler. train() is a no-op: there is nothing to learn,
the feedforward inversion is derived directly from the plant's own
equations at construction time. solve() runs the (near-instant) batch
computation for a given, fixed activation pattern.

Tracking vs. scoring target: the exact inversion is driven to reproduce the
SMOOTHED, physically-achievable target (build_targets()'s smoothed
variant), but solve()/evaluate()/plotting report against the UNSMOOTHED
idealized square-wave target -- matching the same convention used by RL.

Unit boundary: Policy/physics.py work in ns (matching ModulatorArray's
native units). This file's state-space math (matching FF_Decoupler)
requires SI units (seconds, Henries, Farads), so the conversion happens
once here, at __init__ (self._times_s), rather than anywhere upstream.
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh, expm
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.lines import Line2D

from policy import Policy, EnvironmentalConditions, BRAND_PURPLE, BRAND_BLACK


@njit(cache=True)
def _propagate_exact_uniform_jit(Ad, Bd0, Bd1, u_traj, x0):
    """Compiled per-step recursion for _propagate_exact on a uniform time
    grid: x[:,k+1] = Ad@x[:,k] + Bd0@u[:,k] + Bd1@u[:,k+1]."""
    n = Ad.shape[0]
    T = u_traj.shape[1]
    x = np.empty((n, T))
    x[:, 0] = x0
    for k in range(T - 1):
        x[:, k + 1] = Ad @ x[:, k] + Bd0 @ u_traj[:, k] + Bd1 @ u_traj[:, k + 1]
    return x


class ClassicalControl(Policy):
    def __init__(self,
                 conditions: EnvironmentalConditions,
                 dr: float | npt.NDArray[np.float64] = 0.2,
                 wn: float | npt.NDArray[np.float64] | None = None,
                 plant_mismatch: float = 0.0,
                 plant_mismatch_mode: str = "per_entry",
                 rng: int | np.random.Generator | None = None):
        """
        dr, wn: desired reference-model damping ratio / natural frequency
            per channel (wn=None -> auto-matched to each channel's
            dominant physical mode).
        plant_mismatch, plant_mismatch_mode, rng: classical-only robustness
            testing feature (unchanged from FF_Decoupler) -- no RL
            equivalent, defaults to 0.0 (no mismatch, i.e. off).
        """
        super().__init__(conditions)
        if plant_mismatch_mode not in ("per_entry", "global"):
            raise ValueError(f"plant_mismatch_mode must be 'per_entry' or 'global', got {plant_mismatch_mode!r}.")
        self.dr = dr
        self.wn = wn
        self.plant_mismatch = plant_mismatch
        self.plant_mismatch_mode = plant_mismatch_mode
        self._rng = np.random.default_rng(rng)
        self._times_s = self.times * 1e-9  # ns -> s, matches L/C/R below

    # ---- state-space construction ----

    def _perturb_nonzero_symmetric(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Relative multiplicative perturbation of each nonzero (symmetric)
        entry of M, for plant_mismatch robustness testing. See
        plant_mismatch's docstring in __init__ for the full rationale."""
        if self.plant_mismatch <= 0:
            return M.copy()
        n = M.shape[0]
        M_pert = M.copy()
        global_eps = None
        if self.plant_mismatch_mode == "global":
            global_eps = self._rng.uniform(-self.plant_mismatch, self.plant_mismatch)
        for i in range(n):
            for j in range(i, n):
                if M[i, j] != 0:
                    eps = global_eps if global_eps is not None else \
                        self._rng.uniform(-self.plant_mismatch, self.plant_mismatch)
                    M_pert[i, j] = M[i, j] * (1 + eps)
                    M_pert[j, i] = M_pert[i, j]
        return M_pert

    def _construct_state_space(self) -> None:
        """
        Builds every matrix needed for the two-stage feedforward inversion:
        A/B (the true, possibly-mismatched plant), A_D/B_D (decoupled
        reference model), C_D/D_D (LC-stage inversion), A_pd/B_pd
        (bias-tee inversion). Identical math to FF_Decoupler.construct_state_space.
        """
        mod_info = self.mod_array.describe(include_matrices=True)
        n = self.conditions.num_modulators

        # Nominal RLC matrices -- the decoupler's own model of the plant,
        # used for mode analysis and the predistortion inverse regardless
        # of plant_mismatch.
        self.L = mod_info["matrices"]["l_mat"] * 1e-9   # nH -> H
        self.Linv = np.linalg.inv(self.L)
        self.R = mod_info["matrices"]["r2_mat"]
        self.C = mod_info["matrices"]["c_mod_mat"] * 1e-9  # nF -> F
        self.Cinv = np.linalg.inv(self.C)

        # True plant RLC matrices -- identical to nominal unless
        # plant_mismatch > 0. Only these feed into A, B (the actual
        # propagated dynamics); the decoupler itself never sees them.
        self.L_plant = self._perturb_nonzero_symmetric(self.L)
        self.Linv_plant = np.linalg.inv(self.L_plant)
        self.R_plant = self._perturb_nonzero_symmetric(self.R)
        self.C_plant = self._perturb_nonzero_symmetric(self.C)
        self.Cinv_plant = np.linalg.inv(self.C_plant)

        # Physical normal modes of the (nominal) coupled LC network.
        w2, self.Phi = eigh(self.Cinv, self.L)
        w2 = np.clip(w2, 0, None)
        self.w_phys = np.sqrt(w2)

        if self.wn is None:
            self.dominant_mode = np.argmax(np.abs(self.Phi), axis=1)
            self.wn_resolved = self.w_phys[self.dominant_mode]
        else:
            wn_arr = np.atleast_1d(np.asarray(self.wn, dtype=np.float64))
            self.wn_resolved = wn_arr * np.ones(n) if wn_arr.size == 1 else wn_arr
            self.dominant_mode = None

        dr_arr = np.atleast_1d(np.asarray(self.dr, dtype=np.float64))
        self.dr_resolved = dr_arr * np.ones(n) if dr_arr.size == 1 else dr_arr

        self.omega = np.diag(self.wn_resolved)
        self.Z = np.diag(self.dr_resolved)
        self.I_n = np.eye(n)

        # True plant dynamics (possibly mismatched).
        self.A = np.zeros((2 * n, 2 * n))
        self.A[:n, n:] = self.Cinv_plant
        self.A[n:, :n] = -self.Linv_plant
        self.A[n:, n:] = -self.Linv_plant @ self.R_plant
        self.B = np.zeros((2 * n, n))
        self.B[n:, :] = self.Linv_plant

        # Decoupled reference model.
        self.A_D = np.zeros((2 * n, 2 * n))
        self.A_D[:n, n:] = self.I_n
        self.A_D[n:, :n] = -self.omega ** 2
        self.A_D[n:, n:] = -2 * self.Z @ self.omega
        self.B_D = np.zeros((2 * n, n))
        self.B_D[n:, :] = self.omega ** 2

        # LC-stage feedforward inversion (uses the NOMINAL L, C -- the
        # decoupler's own model, not the possibly-mismatched plant).
        LC_omega2 = self.L @ self.C @ (self.omega ** 2)
        self.C_D = np.zeros((n, 2 * n))
        self.C_D[:, :n] = self.I_n - LC_omega2
        self.C_D[:, n:] = self.R @ self.C - 2 * self.L @ self.C @ self.Z @ self.omega
        self.D_D = LC_omega2

        # Bias-tee (droop) inversion.
        self.rc = mod_info["derived"]["rc"] * 1e-9
        self.rc_inv = 1 / self.rc
        self.skip_cap = mod_info["parameters"]["skip_cap"]
        self.A_pd = np.zeros((n, n))
        self.B_pd = self.rc_inv * self.I_n

        self.BC_D = self.B @ self.C_D
        self.BD_D = self.B @ self.D_D

    # ---- exact discretization / propagation ----

    @staticmethod
    def _exact_discretize(A, B, h):
        """Van Loan's method: exact discretization of x'=Ax+Bu(tau), u
        linearly interpolated between samples. Returns (Ad, Bd0, Bd1).

        Minimal augmented state [x, v, w] (size n+2m, not n+3m): v carries
        the term that feeds x via B, w is a constant generator with v'=w.
        Seeding v(0)=I (w(0)=0) holds v constant -> the flat/ZOH integral;
        seeding w(0)=I (v(0)=0) makes v(tau)=tau*I -> the ramp/FOH integral.
        """
        n, m = B.shape
        M = np.zeros((n + 2 * m, n + 2 * m))
        M[:n, :n] = A
        M[:n, n:n + m] = B
        M[n:n + m, n + m:] = np.eye(m)
        Phi = expm(M * h)
        Ad = Phi[:n, :n]
        Phi_xv = Phi[:n, n:n + m]
        Phi_xw = Phi[:n, n + m:]
        Bd0 = Phi_xv - Phi_xw / h
        Bd1 = Phi_xw / h
        return Ad, Bd0, Bd1

    @classmethod
    def _propagate_exact(cls, A, B, times_s, u_traj, x0):
        """Exact propagation of x'=Ax+Bu(t) across times_s (seconds), u_traj
        piecewise-linear between samples."""
        n = A.shape[0]
        dt = np.diff(times_s)

        if len(dt) > 0 and np.allclose(dt, dt[0], atol=0):
            Ad, Bd0, Bd1 = cls._exact_discretize(A, B, dt[0])
            return _propagate_exact_uniform_jit(
                np.ascontiguousarray(Ad), np.ascontiguousarray(Bd0),
                np.ascontiguousarray(Bd1), np.ascontiguousarray(u_traj, dtype=np.float64),
                np.ascontiguousarray(x0, dtype=np.float64),
            )

        x = np.zeros((n, len(times_s)))
        x[:, 0] = x0
        cache = {}
        for k in range(len(times_s) - 1):
            h = times_s[k + 1] - times_s[k]
            key = round(h, 15)
            if key not in cache:
                cache[key] = cls._exact_discretize(A, B, h)
            Ad, Bd0, Bd1 = cache[key]
            x[:, k + 1] = Ad @ x[:, k] + Bd0 @ u_traj[:, k] + Bd1 @ u_traj[:, k + 1]
        return x

    # ---- the two solve modes: predistorted and raw baseline ----
    # Both are driven by whatever target_input is passed in -- callers pass
    # the SMOOTHED target_input (what the inversion tracks); scoring against
    # the unsmoothed target happens separately, using the returned v_mod.

    def _run_full_predistorted_exact(self, target_input: npt.NDArray[np.float64]):
        """
        Full two-stage predistortion (bias-tee + LC-crosstalk inversion).
        Returns (v_action, v_load, x_D, x_plant). If skip_cap, reduces to
        the LC-only inversion with v_action == v_load.
        """
        n = self.conditions.num_modulators

        if self.skip_cap:
            Abig = np.zeros((4 * n, 4 * n))
            Abig[:2 * n, :2 * n] = self.A_D
            Abig[2 * n:, :2 * n] = self.BC_D
            Abig[2 * n:, 2 * n:] = self.A
            Bbig = np.zeros((4 * n, n))
            Bbig[:2 * n, :] = self.B_D
            Bbig[2 * n:, :] = self.BD_D
            x0_big = np.zeros(4 * n)
            x_big = self._propagate_exact(Abig, Bbig, self._times_s, target_input, x0_big)
            x_D = x_big[:2 * n, :]
            x_plant = x_big[2 * n:, :]
            u_out = self.C_D @ x_D + self.D_D @ target_input
            return u_out, u_out, x_D, x_plant

        rc_inv = self.rc_inv
        Afull = np.zeros((6 * n, 6 * n))
        Bfull = np.zeros((6 * n, n))

        Afull[:2*n, :2*n] = self.A_D
        Bfull[:2*n, :] = self.B_D

        Afull[2*n:3*n, :2*n] = rc_inv * self.C_D
        Bfull[2*n:3*n, :] = rc_inv * self.D_D

        Afull[3*n:4*n, :2*n] = rc_inv * self.C_D
        Afull[3*n:4*n, 2*n:3*n] = rc_inv * self.I_n
        Afull[3*n:4*n, 3*n:4*n] = -rc_inv * self.I_n
        Bfull[3*n:4*n, :] = rc_inv * self.D_D

        Afull[4*n:6*n, :2*n] = self.BC_D
        Afull[4*n:6*n, 2*n:3*n] = self.B
        Afull[4*n:6*n, 3*n:4*n] = -self.B
        Afull[4*n:6*n, 4*n:6*n] = self.A
        Bfull[4*n:6*n, :] = self.BD_D

        x0_big = np.zeros(6 * n)
        x_big = self._propagate_exact(Afull, Bfull, self._times_s, target_input, x0_big)

        x_D = x_big[:2*n, :]
        v_cap_pd = x_big[2*n:3*n, :]
        v_cap_bt = x_big[3*n:4*n, :]
        x_plant = x_big[4*n:6*n, :]

        v_load_des = self.C_D @ x_D + self.D_D @ target_input
        v_action = v_load_des + v_cap_pd
        v_load = v_action - v_cap_bt
        return v_action, v_load, x_D, x_plant

    def _run_full_raw_exact(self, target_input: npt.NDArray[np.float64]):
        """Baseline, no predistortion: target_input fed directly as
        v_action through the real bias-tee + LC stages. Returns (v_load, x_plant)."""
        n = self.conditions.num_modulators

        if self.skip_cap:
            x_plant = self._propagate_exact(self.A, self.B, self._times_s, target_input, np.zeros(2 * n))
            return target_input, x_plant

        rc_inv = self.rc_inv
        Afull = np.zeros((3 * n, 3 * n))
        Bfull = np.zeros((3 * n, n))
        Afull[:n, :n] = -rc_inv * self.I_n
        Bfull[:n, :] = rc_inv * self.I_n
        Afull[n:3*n, :n] = -self.B
        Afull[n:3*n, n:3*n] = self.A
        Bfull[n:3*n, :] = self.B

        x0_big = np.zeros(3 * n)
        x_big = self._propagate_exact(Afull, Bfull, self._times_s, target_input, x0_big)
        v_cap_bt_raw = x_big[:n, :]
        x_plant_raw = x_big[n:3*n, :]
        v_load_raw = target_input - v_cap_bt_raw
        return v_load_raw, x_plant_raw

    # ---- Policy interface ----

    def train(self, **kwargs) -> None:
        """No-op: classical control has no training phase."""
        return None

    def solve(self, active_channels: list[int] | None = None):
        """
        Runs the full two-stage predistortion, driven to track the
        SMOOTHED target, for the given fixed activation pattern. Returns
        (v_mod_hist, v_src, active, target_powers) where target_powers is
        the UNSMOOTHED idealized target -- per the Policy.solve contract,
        this is what final scoring compares against, not what the inversion
        was tracking internally.
        """
        active = self.build_active_mask(active_channels)
        (target_powers_s, target_input_s,
         target_powers_u, target_input_u) = self.build_targets(active)
        self._construct_state_space()
        v_action, v_load, x_D, x_plant = self._run_full_predistorted_exact(target_input_s)
        n = self.conditions.num_modulators
        v_mod_hist = x_plant[:n, :]
        return v_mod_hist, v_action, active, target_powers_u

    # ---- classical-only extra: baseline comparison plot ----

    def plot_with_baseline(self, active_channels: list[int] | None = None, show_raw: bool = True):
        """
        Classical-only four-panel comparison plot (input voltage / voltage
        at load / optical power vs. target / residual), paginated per
        channel. Both the predistorted and raw-baseline runs are driven by
        the SMOOTHED target_input; the plotted dashed reference line is
        the UNSMOOTHED idealized target, matching final scoring's
        convention. Not part of the shared Policy.plot() since RL has no
        equivalent "no compensation" baseline to show alongside its
        result.

        show_raw: if False, skips the non-predistorted (raw) baseline
            entirely -- both its computation (_run_full_raw_exact) and its
            traces/legend entry -- leaving just target vs. predistorted.
        """
        active = self.build_active_mask(active_channels)
        (target_powers_s, target_input_s,
         target_powers_u, target_input_u) = self.build_targets(active)
        self._construct_state_space()

        v_action, v_load, x_D, x_plant = self._run_full_predistorted_exact(target_input_s)

        n = self.conditions.num_modulators
        v_mod_pre = x_plant[:n, :]
        power_pre = self.mod_array.calc_power(v_mod_pre)
        p_residual_pre = power_pre - target_powers_u

        if show_raw:
            v_load_raw, x_plant_raw = self._run_full_raw_exact(target_input_s)
            v_mod_raw = x_plant_raw[:n, :]
            power_raw = self.mod_array.calc_power(v_mod_raw)
            p_residual_raw = power_raw - target_powers_u

        fig, (ax_in, ax_vload, ax_pow, ax_resid) = plt.subplots(1, 4, figsize=(18, 4.2))
        plt.subplots_adjust(bottom=0.28, top=0.8, wspace=0.45)

        legend_handles = [Line2D([0], [0], color=BRAND_BLACK, linestyle="--", linewidth=2, label="target (unsmoothed)")]
        if show_raw:
            legend_handles.append(Line2D([0], [0], color=BRAND_PURPLE, alpha=0.85, label="non-predistorted"))
        legend_handles.append(Line2D([0], [0], color=BRAND_PURPLE, label="predistorted"))
        fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=9,
                   bbox_to_anchor=(0.5, 0.12))

        state = {"i": 0}

        def draw_channel(i):
            for ax in (ax_in, ax_vload, ax_pow, ax_resid):
                ax.clear()
                ax.tick_params(axis="y", labelsize=8)

            if show_raw:
                ax_in.plot(self.times, target_input_u[i], color=BRAND_PURPLE, alpha=0.85)
            ax_in.plot(self.times, v_action[i], color=BRAND_PURPLE)
            ax_in.set_title("Input voltage (command)", pad=15)
            ax_in.set_xlabel("time (ns)")
            ax_in.set_ylabel("V")

            if show_raw:
                ax_vload.plot(self.times, v_mod_raw[i], color=BRAND_PURPLE, alpha=0.85)
            ax_vload.plot(self.times, v_mod_pre[i], color=BRAND_PURPLE)
            ax_vload.set_title("Voltage at load", pad=15)
            ax_vload.set_xlabel("time (ns)")
            ax_vload.set_ylabel("V")

            ax_pow.plot(self.times, target_powers_u[i], color=BRAND_BLACK, linestyle="--", linewidth=2)
            if show_raw:
                ax_pow.plot(self.times, power_raw[i], color=BRAND_PURPLE, alpha=0.85)
            ax_pow.plot(self.times, power_pre[i], color=BRAND_PURPLE)
            ax_pow.set_title("Optical power vs. target", pad=15)
            ax_pow.set_xlabel("time (ns)")
            ax_pow.set_ylabel("nW")

            ax_resid.axhline(0, color=BRAND_BLACK, linestyle="--", linewidth=2)
            if show_raw:
                ax_resid.scatter(self.times, p_residual_raw[i], color=BRAND_PURPLE, alpha=0.6, s=5)
            ax_resid.scatter(self.times, p_residual_pre[i], color=BRAND_PURPLE, alpha=0.6, s=5)
            ax_resid.set_title("Optical power residual vs. target", pad=15)
            ax_resid.set_xlabel("time (ns)")
            ax_resid.set_ylabel("nW")

            title = f"Channel {i + 1} of {n}: predistorted vs. non-predistorted" if show_raw \
                else f"Channel {i + 1} of {n}: predistorted"
            fig.suptitle(title, y=0.92)
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