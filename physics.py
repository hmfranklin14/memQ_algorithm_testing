"""
physics.py

Shared physics engine for the modulator array. Owned by neither the
classical control nor RL approach -- both import this module directly and
treat it as ground truth.

This is your partner's mod_phys.py, unchanged except for one line in the
JIT reward kernel: the active/inactive worst-case (max) squared error split
has been replaced with a plain mean over all channels, for consistency with
the RL side's reward convention (per decision: the max-based reward was an
earlier experiment that didn't improve performance, so both approaches now
use -mean(sq_err) everywhere). active_mask/skip_cap are kept in the
_physics_step/step signatures for compatibility even though active_mask no
longer affects the reward.
"""

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def _physics_step(v_action, v_cap, v_mod, i_ind,
                   a_diag, m_diag, g_diag, h,
                   l_mat, c_mod_mat, big2_inv,
                   p_in, null_pt, dc_bias, vpi_val, er_val, target_val,
                   active_mask, skip_cap):
    n = v_action.shape[0]

    if skip_cap:
        # Bias-tee stage bypassed entirely: v_src passes through to the
        # load unmodified, and there's no capacitor voltage to track.
        v_cap_new = v_cap
        v_load = v_action
    else:
        # Bias-tee stage is exactly diagonal (no coupling between
        # channels), so this is an elementwise O(n) update rather than a
        # matrix solve -- a_diag/m_diag/g_diag are the diagonals of
        # a_inv/m_mat/g_mat.
        v_cap_new = a_diag * (m_diag * v_cap + h * (g_diag * v_action))
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

    # Changed from active/inactive max-squared-error split to a plain mean
    # over all channels -- matches RL's existing reward convention. The
    # active_mask parameter is kept (unused) for signature compatibility.
    reward = -np.mean(sq_err)

    return v_cap_new, v_mod_new, i_ind_new, power, reward


class ModulatorArray:
    """
    Encapsulates the physics of a 1D array of optical modulators.

    Circuit stages:
      - Bias-tee stage (t_res, cap_t): purely per-channel, no coupling
        between modulators
      - Inductive / modulator-cap stage: series inductance with
        nearest-neighbor mutual coupling (ind, ind_p) and shunt modulator
        capacitance with nearest-neighbor parasitic capacitive coupling
        (mod_cap, cap_p), plus series modulator resistance (mod_res)

    Electro-optic response:
      - Sinusoidal power transfer function parameterized by v_pi, null_pt,
        p_in (input power) and er (extinction ratio)

    The class owns the dynamical state (capacitor voltage, modulator
    voltage, inductor current) and advances it one timestep at a time via
    `step`.
    """

    def __init__(self,
                 num_modulators: int,
                 h: float,
                 rf_amp: float = 1.3,
                 dc_bias: float = 0.8,
                 v_pi: float = 1.3,
                 null_pt: float = 0.8,
                 p_in: float = 1000,
                 er: float = 29.6,
                 t_res: float = 50,
                 mod_res: float = 50,
                 cap_t: float = 50,
                 cap_p: float = 0.0001,
                 mod_cap: float = 0.0003,
                 ind: float = 2.7,
                 ind_p: float = 1,
                 skip_cap: bool = False):
        '''
        Parameters:
            num_modulators: Int, number of modulators in the array
            h: Float, simulation timestep in ns (fixes the discretized
                circuit matrices, so it must be known at construction time)
            rf_amp, dc_bias: Floats representing peak RF and DC bias
                voltages for each modulator in V
            v_pi: Float representing the half-wave voltage for each
                modulator in V
            null_pt: Float representing the minimum null voltage for each
                modulator in V
            p_in: Float representing the input power for each modulator in nW
            er: Float representing the extinction ratio for each modulator
                in dB
            t_res: Float representing the characteristic system resistance
                in the bias-tee circuit in Ohms
            mod_res: Float representing the resistance at the modulator in
                Ohms
            cap_t: Float representing the capacitance of the bias-tee
                capacitor in nF
            cap_p: Float representing the parasitic capacitance between
                adjacent transmission lines in nF (couples nearest-neighbor
                modulator capacitances in the inductive/modulator-cap
                stage; the bias-tee stage has no coupling)
            mod_cap: Float representing the capacitance of the modulator
                electrodes in nF
            ind: Float representing the inherent inductance of the
                transmission lines in nH
            ind_p: Float representing the parasitic inductance between
                adjacent transmission lines in nH
            skip_cap: Bool. If True, the bias-tee stage is bypassed
                entirely -- v_src passes through to the load unmodified
                (v_load = v_action) instead of having the bias-tee
                capacitor voltage subtracted from it, and the bias-tee
                capacitor voltage (v_cap) is never updated.
        '''
        self.num_modulators = num_modulators
        self.h = h
        self.rf_amp = rf_amp
        self.dc_bias = dc_bias
        self.v_pi = v_pi
        self.null_pt = null_pt
        self.p_in = p_in
        self.er = er
        self.t_res = t_res
        self.mod_res = mod_res
        self.cap_t = cap_t
        self.cap_p = cap_p
        self.mod_cap = mod_cap
        self.ind = ind
        self.ind_p = ind_p
        self.skip_cap = skip_cap

        self.p_min = self.p_in / 10 ** (self.er / 10)
        self.rc = self.t_res * self.cap_t

        self._build_matrices()
        self.reset_state()

    def _build_matrices(self):
        n = self.num_modulators

        # --- Bias-tee stage: purely per-channel RC, no coupling ---
        self.g_mat = np.diag([1 / self.t_res] * n)
        self.m_mat = np.zeros((n, n))
        np.fill_diagonal(self.m_mat, self.cap_t)
        self.a_inv = np.linalg.inv(self.m_mat + self.h * self.g_mat)

        # 1D diagonals of the (purely diagonal) bias-tee matrices, used by
        # the hot-path step() as an O(n) elementwise update instead of an
        # O(n^2) matrix-vector solve. self.g_mat/m_mat/a_inv above are kept
        # as full matrices for inspection (describe()) and any future
        # linear-control-theory use (e.g. LQR/MPC state-space extraction).
        self._g_diag = np.diag(self.g_mat).copy()
        self._m_diag = np.diag(self.m_mat).copy()
        self._a_diag = np.diag(self.a_inv).copy()

        # --- Inductive / modulator-cap stage: L (self + mutual, ind/ind_p)
        #     + series R2 (mod_res) + shunt C_mod (self + mutual,
        #     mod_cap/cap_p), before the modulator input ---
        self.l_mat = np.zeros((n, n))
        np.fill_diagonal(self.l_mat, self.ind)
        for i in range(n - 1):
            self.l_mat[i][i + 1] += self.ind_p
            self.l_mat[i + 1][i] += self.ind_p

        self.r2_mat = np.diag([self.mod_res] * n)
        # Nodal (Maxwell) capacitance matrix for the shunt C_mod stage: each
        # cap_p is a real capacitor wired node-to-node between neighbors, so
        # KCL (i_ind = c_mod_mat @ dv_mod/dt) requires the graph-Laplacian
        # form -- diagonal picks up +cap_p per adjoining neighbor, and the
        # off-diagonal coupling term is -cap_p (current flows out of node i
        # into the coupler when v_i > v_j)
        self.c_mod_mat = np.zeros((n, n))
        np.fill_diagonal(self.c_mod_mat, self.mod_cap)
        for i in range(n - 1):
            self.c_mod_mat[i][i] += self.cap_p
            self.c_mod_mat[i + 1][i + 1] += self.cap_p
            self.c_mod_mat[i][i + 1] -= self.cap_p
            self.c_mod_mat[i + 1][i] -= self.cap_p

        big2 = np.zeros((2 * n, 2 * n))
        big2[:n, :n] = self.c_mod_mat
        big2[:n, n:] = -self.h * np.eye(n)
        big2[n:, :n] = self.h * np.eye(n)
        big2[n:, n:] = self.l_mat + self.h * self.r2_mat
        self.big2_inv = np.linalg.inv(big2)

    def reset_state(self):
        """Zero out the dynamical state (capacitor / modulator voltages,
        inductor current, and the cheap capacitor-voltage estimate)."""
        n = self.num_modulators
        self._v_cap = np.zeros(n, dtype=np.float64)
        self._v_mod = np.zeros(n, dtype=np.float64)
        self._i_ind = np.zeros(n, dtype=np.float64)
        self._v_cap_estimate = np.zeros(n, dtype=np.float64)

    def update_params(self, **kwargs) -> None:
        """
        Update one or more circuit/electro-optic parameters in place and
        rebuild the discretized matrices to match. Accepts any of the
        constructor's keyword parameters (rf_amp, dc_bias, v_pi, null_pt,
        p_in, er, t_res, mod_res, cap_t, cap_p, mod_cap, ind, ind_p,
        skip_cap).

        Lets the same ModulatorArray instance be reused across a parameter
        sweep (e.g. for interactive plotting) instead of constructing a
        new one every time. Does NOT touch the dynamical state (v_cap,
        v_mod, i_ind, v_cap_estimate) or the timestep h -- call
        reset_state() separately (or just call an owning ModulatorEnv's
        reset()) if you want those cleared before the next step().
        """
        valid_params = {
            "rf_amp", "dc_bias", "v_pi", "null_pt", "p_in", "er",
            "t_res", "mod_res", "cap_t", "cap_p", "mod_cap", "ind", "ind_p",
            "skip_cap",
        }
        unknown = set(kwargs) - valid_params
        if unknown:
            raise TypeError(
                f"update_params() got unexpected parameter(s): {sorted(unknown)}")

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.p_min = self.p_in / 10 ** (self.er / 10)
        self.rc = self.t_res * self.cap_t
        self._build_matrices()

    @property
    def v_cap(self) -> npt.NDArray[np.float64]:
        return self._v_cap

    @property
    def v_mod(self) -> npt.NDArray[np.float64]:
        return self._v_mod

    @property
    def i_ind(self) -> npt.NDArray[np.float64]:
        return self._i_ind

    @property
    def v_cap_estimate(self) -> npt.NDArray[np.float64]:
        return self._v_cap_estimate

    def update_v_cap_estimate(self, v_action: npt.NDArray[np.float64]) -> None:
        """
        Cheap first-order (single-pole) estimate of the bias-tee capacitor
        voltage. This is intentionally simpler than the full coupled
        capacitive-stage solve used in `step`, and is meant to be exposed
        to an observer/agent as a rough proxy signal.

        No-op when skip_cap is True -- with the bias-tee bypassed, there's
        no capacitor voltage to estimate, so v_cap_estimate stays at 0.
        """
        if self.skip_cap:
            return
        self._v_cap_estimate = self._v_cap_estimate + (
            self.h / self.rc) * (v_action - self._v_cap_estimate)

    def describe(self, include_matrices: bool = True) -> dict:
        """
        Returns a dictionary summarizing the full modulator-array setup:
        construction parameters, derived quantities, the discretized
        circuit matrices, and the current dynamical state.

        Parameters:
            include_matrices: If False, omits the "matrices" section (they
                grow as O(num_modulators^2) and can be large/unwieldy to
                print for big arrays).
        """
        info = {
            "num_modulators": self.num_modulators,
            "timestep_h": self.h,
            "parameters": {
                "rf_amp": self.rf_amp,
                "dc_bias": self.dc_bias,
                "v_pi": self.v_pi,
                "null_pt": self.null_pt,
                "p_in": self.p_in,
                "er": self.er,
                "t_res": self.t_res,
                "mod_res": self.mod_res,
                "cap_t": self.cap_t,
                "cap_p": self.cap_p,
                "mod_cap": self.mod_cap,
                "ind": self.ind,
                "ind_p": self.ind_p,
                "skip_cap": self.skip_cap,
            },
            "derived": {
                "p_min": self.p_min,
                "rc": self.rc,
            },
            "state": {
                "v_cap": self._v_cap.copy(),
                "v_mod": self._v_mod.copy(),
                "i_ind": self._i_ind.copy(),
                "v_cap_estimate": self._v_cap_estimate.copy(),
            },
        }

        if include_matrices:
            info["matrices"] = {
                "g_mat": self.g_mat.copy(),
                "m_mat": self.m_mat.copy(),
                "a_inv": self.a_inv.copy(),
                "l_mat": self.l_mat.copy(),
                "r2_mat": self.r2_mat.copy(),
                "c_mod_mat": self.c_mod_mat.copy(),
                "big2_inv": self.big2_inv.copy(),
            }

        return info

    def calc_power(self,
                    v_load: npt.NDArray[np.float64],
                    vpi_val: npt.NDArray[np.float64] | float | None = None,
                    er_val: npt.NDArray[np.float64] | float | None = None
                    ) -> npt.NDArray[np.float64]:
        """
        Calculates the output optical power of each modulator given the
        voltage at the load. `vpi_val`/`er_val` allow per-call overrides
        (e.g. randomized per-timestep values); they default to the
        nominal `v_pi`/`er` this array was constructed with.
        """
        vpi_val = self.v_pi if vpi_val is None else vpi_val
        er_val = self.er if er_val is None else er_val
        p_min = self.p_in / 10 ** (er_val / 10)
        return p_min + (self.p_in - p_min) * np.sin(
            (v_load + self.dc_bias - self.null_pt) * np.pi / (2 * vpi_val)
        ) ** 2

    def calc_power_inv(self,
                    powers: npt.NDArray[np.float64],
                    vpi_val: npt.NDArray[np.float64] | float | None = None,
                    er_val: npt.NDArray[np.float64] | float | None = None
                    ) -> npt.NDArray[np.float64]:
        vpi_val = self.v_pi if vpi_val is None else vpi_val
        er_val = self.er if er_val is None else er_val
        p_min = self.p_in / 10 ** (er_val / 10)
        return (self.null_pt - self.dc_bias + 2*vpi_val/np.pi
                * np.arcsin(np.sqrt((powers-p_min)/(self.p_in-p_min))))

    def calc_infidelity(self,
                          v_mod_hist: npt.NDArray[np.float64],
                          target_power: npt.NDArray[np.float64],
                          vpi_val: npt.NDArray[np.float64] | float | None = None,
                          er_val: npt.NDArray[np.float64] | float | None = None
                          ) -> npt.NDArray[np.float64]:
        """
        Returns an array of length num_modulators containing the pulse
        infidelity for each modulator, given the full-episode modulator
        voltage history and target power trace (both channel-major, shape
        (n, T)). Goal infidelity is 0.001 per modulator.

        Both approaches call this with the SMOOTHED target_powers from
        Policy.build_targets(), not an unsmoothed idealized square wave.

        Note: this is currently the infidelity for strictly NOT gates. To
        generalize, replace np.pi with some other target phase, either
        globally or on a modulator-by-modulator basis.
        """
        vpi_val = self.v_pi if vpi_val is None else vpi_val
        er_val = self.er if er_val is None else er_val
        all_powers = self.calc_power(v_mod_hist, vpi_val=vpi_val, er_val=er_val)
        area_ratio = (np.trapezoid(np.sqrt(all_powers), axis=1)
                      / np.trapezoid(np.sqrt(target_power), axis=1))
        return np.sin(np.pi * (area_ratio - 1) / 2) ** 2

    def step(self,
              v_action: npt.NDArray[np.float64],
              vpi_val: npt.NDArray[np.float64],
              er_val: npt.NDArray[np.float64],
              target_val: npt.NDArray[np.float64],
              active_mask: npt.NDArray[np.bool_]):
        """
        Advances the modulator-array physics by one timestep `h`, given
        the source voltage applied at this timestep (`v_action`), the
        (possibly randomized) v_pi/er values active at this timestep, the
        target optical power for each modulator, and a boolean mask of
        which modulators are meant to be "active" (kept for signature
        compatibility; no longer affects the reward -- see _physics_step).

        Updates the internal v_cap / v_mod / i_ind state in place and
        returns (power, reward) for this timestep.
        """
        (self._v_cap, self._v_mod, self._i_ind,
         power, reward) = _physics_step(
            v_action, self._v_cap, self._v_mod, self._i_ind,
            self._a_diag, self._m_diag, self._g_diag, self.h,
            self.l_mat, self.c_mod_mat, self.big2_inv,
            self.p_in, self.null_pt, self.dc_bias, vpi_val, er_val,
            target_val, active_mask, self.skip_cap,
        )
        return power, reward