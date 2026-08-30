"""
evaluate.py

eval() lets a user specify environmental conditions, pick
an approach (classical control, or one of three RL approaches), specify a
fixed activation pattern to test against, and get back the same kind of
output regardless of approach: paginated per-channel plots (three panels:
input voltage, power vs. target, residual), per-channel + overall MSE
(raw/masked/transition), per-channel infidelity.

For RL approaches, checks a model cache (keyed on approach + that
approach's own POLICY_VERSION + environmental conditions + RL
hyperparameters) before training from scratch, so re-running the same
conditions/approach to inspect a different activation pattern doesn't
require retraining every time. Classical control has no training phase,
so it's never cached.

sweep_active_configurations() sweeps every non-empty active-channel
arrangement for a given approach, grouping results by active-modulator
count, and writes avg/std MSE/infidelity/masked-MSE/transition-MSE to
CSV. For RL approaches, ONE policy is trained/loaded and reused
deterministically across every arrangement; for classical, optionally
re-drawn across plant_mismatch trials per arrangement.

IMPORTANT -- conditions.h: this must be set by the caller to match
total_time/(num_points-1) for the given pre_pad/pulse_width/post_pad/
num_pulses/inter_pulse_gap -- Policy.evaluate()'s masked/transition-MSE
windowing uses conditions.h directly (not derived from the actual time
grid), so a mismatched h silently produces incorrect window boundaries.
"""

import csv
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import numpy as np

from policy import Policy, EnvironmentalConditions
from classical_control import ClassicalControl
from rl_v_cap_estimate import RLVCapEstimate, POLICY_VERSION as V_CAP_ESTIMATE_POLICY_VERSION
from rl_standard import RLStandard, POLICY_VERSION as STANDARD_RL_POLICY_VERSION
from rl_scaled import RLScaled, POLICY_VERSION as SCALED_RL_POLICY_VERSION

# ==== Environment configuration -- edit these to define what to train/evaluate on ====
NUM_MODULATORS = 5
NUM_PULSES = 1
PULSE_WIDTH = 150.0          # ns
INTER_PULSE_GAP = 1000.0     # ns
PRE_PAD = 25.0                # ns
POST_PAD = 25.0                # ns
NUM_POINTS = 2000
RISE_TIME_NS = 0
TRANSITION_WINDOW_NS = 10.0
H = 0.1

RF_AMP = 1.3
DC_BIAS = 0.8
V_PI = 1.3
NULL_PT = 0.8
P_IN = 1000
ER = 29.6
T_RES = 50
MOD_RES = 50
CAP_T = 50
CAP_P = 0.0001
MOD_CAP = 0.0003
IND = 2.7
IND_P = 1
SKIP_CAP = False

CACHE_DIR = "./trained_models"

# ---- approach registry ----
APPROACH_REGISTRY = {
    "classical": {"policy_class": ClassicalControl, "is_rl": False, "policy_version": None},
    "v_cap_estimate": {"policy_class": RLVCapEstimate, "is_rl": True,
                        "policy_version": V_CAP_ESTIMATE_POLICY_VERSION},
    "standard_rl": {"policy_class": RLStandard, "is_rl": True,
                     "policy_version": STANDARD_RL_POLICY_VERSION},
    "scaled_rl": {"policy_class": RLScaled, "is_rl": True,
                  "policy_version": SCALED_RL_POLICY_VERSION},
}

_live_widget_refs = []


def build_default_conditions() -> EnvironmentalConditions:
    return EnvironmentalConditions(
        num_modulators=NUM_MODULATORS,
        h=H,
        num_pulses=NUM_PULSES,
        pulse_width=PULSE_WIDTH,
        inter_pulse_gap=INTER_PULSE_GAP,
        pre_pad=PRE_PAD,
        post_pad=POST_PAD,
        num_points=NUM_POINTS,
        rise_time_ns=RISE_TIME_NS,
        transition_window_ns=TRANSITION_WINDOW_NS,
        rf_amp=RF_AMP,
        dc_bias=DC_BIAS,
        v_pi=V_PI,
        null_pt=NULL_PT,
        p_in=P_IN,
        er=ER,
        t_res=T_RES,
        mod_res=MOD_RES,
        cap_t=CAP_T,
        cap_p=CAP_P,
        mod_cap=MOD_CAP,
        ind=IND,
        ind_p=IND_P,
        skip_cap=SKIP_CAP,
    )


# ---- RL model cache ----

def _config_hash(conditions: EnvironmentalConditions, rl_kwargs: dict) -> str:
    payload = {"conditions": asdict(conditions), "rl_kwargs": rl_kwargs}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_path(approach: str, policy_version: str, config_hash: str) -> str:
    return os.path.join(CACHE_DIR, f"{approach}_{policy_version}_{config_hash}")


def _load_cached_rl(approach: str, conditions: EnvironmentalConditions, rl_kwargs: dict) -> Policy | None:
    entry = APPROACH_REGISTRY[approach]
    policy_version = entry["policy_version"]
    config_hash = _config_hash(conditions, rl_kwargs)
    path = _cache_path(approach, policy_version, config_hash)
    manifest_path = os.path.join(path, "manifest.json")

    if not os.path.isfile(manifest_path):
        return None

    with open(manifest_path) as f:
        manifest = json.load(f)
    if (manifest.get("approach") != approach
            or manifest.get("policy_version") != policy_version
            or manifest.get("config_hash") != config_hash):
        return None

    policy = entry["policy_class"](conditions, **rl_kwargs)
    policy.load(path)
    print(f"Reusing cached {approach} model at {path} (trained {manifest.get('trained_at', 'unknown time')})")
    return policy


def _save_cached_rl(approach: str, policy: Policy, conditions: EnvironmentalConditions, rl_kwargs: dict) -> None:
    entry = APPROACH_REGISTRY[approach]
    policy_version = entry["policy_version"]
    config_hash = _config_hash(conditions, rl_kwargs)
    path = _cache_path(approach, policy_version, config_hash)
    os.makedirs(path, exist_ok=True)
    policy.save(path)

    manifest = {
        "approach": approach,
        "policy_version": policy_version,
        "config_hash": config_hash,
        "conditions": asdict(conditions),
        "rl_kwargs": rl_kwargs,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ---- results reporting ----

def _print_results(results: dict, conditions: EnvironmentalConditions, active_channels) -> None:
    n = conditions.num_modulators
    fired = active_channels if active_channels is not None else list(range(n))
    print(f"\n--- Results ({n} modulators, active channels {fired}) ---")
    print(f"Overall MSE: {results['mse_overall']:.12f}")
    print(f"Overall masked MSE (excludes transition windows): {results['mse_masked_overall']:.12f}")
    print(f"Overall transition-region MSE (fired channels only): {results['mse_transition_overall']:.12f}")
    for m in range(n):
        status = "active" if m in fired else "inactive"
        transition_str = (f"{results['mse_transition_per_channel'][m]:.12f}"
                           if not np.isnan(results['mse_transition_per_channel'][m]) else "N/A (never fires)")
        print(f"  Modulator {m + 1} ({status}): MSE={results['mse_per_channel'][m]:.12f}, "
              f"masked MSE={results['mse_masked_per_channel'][m]:.12f}, "
              f"transition MSE={transition_str}, "
              f"infidelity={results['infidelity_per_channel'][m]:.12f}")
    print(f"Mean infidelity (overall): {results['mean_infidelity_overall']:.12f}")


# ---- the eval() entry point ----

def eval(approach: str,
         conditions: EnvironmentalConditions | None = None,
         active_channels: list[int] | None = None,
         rl_kwargs: dict | None = None,
         classical_kwargs: dict | None = None,
         force_retrain: bool = False,
         plot: bool = True,
         show_baseline: bool = False,
         show_raw: bool = True) -> dict:
    """
    Parameters:
        approach: one of "classical", "v_cap_estimate", "standard_rl",
            "scaled_rl", or "all".
        conditions: EnvironmentalConditions override. If None, built from
            the module-level constants at the top of this file.
        active_channels: which modulators fire, e.g. [0, 1, 2]. None
            means every channel fires.
        rl_kwargs: passed to whichever RL approach's constructor is
            selected -- ignored if approach == "classical".
        classical_kwargs: passed to ClassicalControl's constructor --
            ignored for RL approaches.
        force_retrain: skip the RL model cache and train from scratch.
        plot: whether to render plots.
        show_baseline: classical-only -- if True (and approach=="classical"),
            renders the raw-vs-predistorted baseline comparison plot instead
            of the shared plot().
        show_raw: classical-only, ignored unless show_baseline is also
            True -- if False, drops the non-predistorted (raw) baseline
            trace/legend entry from plot_with_baseline(), leaving just
            target vs. predistorted.

    Returns the dict produced by Policy.evaluate() for a single approach,
    or {"classical": {...}, "v_cap_estimate": {...}, ...} for "all".
    """
    if approach == "all":
        return {
            name: eval(name, conditions, active_channels, rl_kwargs,
                       classical_kwargs, force_retrain, plot, show_baseline, show_raw)
            for name in APPROACH_REGISTRY
        }

    if approach not in APPROACH_REGISTRY:
        raise ValueError(f"Unknown approach: {approach!r} (expected one of "
                          f"{list(APPROACH_REGISTRY)} or 'all')")

    conditions = conditions or build_default_conditions()
    rl_kwargs = rl_kwargs or {}
    classical_kwargs = classical_kwargs or {}
    entry = APPROACH_REGISTRY[approach]

    if not entry["is_rl"]:
        policy = entry["policy_class"](conditions, **classical_kwargs)
        policy.train()  # no-op
    else:
        cached = None if force_retrain else _load_cached_rl(approach, conditions, rl_kwargs)
        if cached is not None:
            policy = cached
        else:
            policy = entry["policy_class"](conditions, **rl_kwargs)
            policy.train()
            _save_cached_rl(approach, policy, conditions, rl_kwargs)

    v_mod_hist, v_src, active, target_powers = policy.solve(active_channels=active_channels)
    results = policy.evaluate(v_mod_hist, active, target_powers)

    _print_results(results, conditions, active_channels)

    if plot:
        if approach == "classical" and show_baseline:
            widget_refs = policy.plot_with_baseline(active_channels=active_channels, show_raw=show_raw)
        else:
            widget_refs = policy.plot(v_src, v_mod_hist, target_powers)
        _live_widget_refs.append(widget_refs)

    return results


# ---- sweep every activation pattern, for any registered approach ----

_SWEEP_METRICS = {
    "mse": "mse_overall",
    "infidelity": "mean_infidelity_overall",
    "mse_masked": "mse_masked_overall",
    "mse_transition": "mse_transition_overall",
}

_SWEEP_DEFAULT_CSV = {
    "classical": "modulator_evaluation_classic.csv",
    "v_cap_estimate": "modulator_evaluation_v_cap_estimate.csv",
    "standard_rl": "modulator_evaluation_standard_rl.csv",
    "scaled_rl": "modulator_evaluation_scaled_rl.csv",
}


def sweep_active_configurations(approach: str = "classical",
                                 conditions: EnvironmentalConditions | None = None,
                                 classical_kwargs: dict | None = None,
                                 rl_kwargs: dict | None = None,
                                 force_retrain: bool = False,
                                 plant_mismatch: float = 0.0,
                                 plant_mismatch_mode: str = "per_entry",
                                 n_mismatch_trials: int = 20,
                                 csv_path: str | None = None) -> np.ndarray:
    """
    Sweeps `approach` over every possible non-empty arrangement of active
    modulators (2**n - 1 arrangements), scoring each with the shared
    Policy.evaluate() metric set: overall MSE, infidelity, masked MSE
    (excludes each channel's transition_window_ns around every rising/
    falling edge), and transition MSE (the complement).

    approach == "classical": reconstructs ClassicalControl once per trial
        (fresh rng draw). Each arrangement is evaluated over
        `n_mismatch_trials` independent plant realizations whenever
        plant_mismatch > 0. With the default plant_mismatch = 0.0 the
        plant is deterministic, so every trial would be identical --
        skipped entirely (1 "trial") to avoid wasted computation.

    approach in {"v_cap_estimate", "standard_rl", "scaled_rl"}: trains (or
        loads, via the same cache eval() uses) ONE RL policy up front and
        reuses it for every arrangement -- retraining per arrangement
        would be enormously wasteful. solve() runs deterministic=True,
        and plant_mismatch has no RL equivalent: passing plant_mismatch >
        0 with an RL approach raises ValueError, and every group's *_std
        column is identically 0.

    Writes one row per num_active_modulators group to `csv_path` (default
    from _SWEEP_DEFAULT_CSV, keyed by approach). For each metric X:
        avg_X:        mean, across arrangements in the group, of that
                       arrangement's own across-trial mean.
        avg_X_std:     mean, across arrangements, of that arrangement's own
                       across-trial std (0 when plant_mismatch=0.0 or
                       approach is an RL approach).
        config_X_std:  std, across arrangements in the group, of each
                       arrangement's across-trial mean.

    Returns the raw per-arrangement (not yet grouped) records as an
    ndarray with columns [num_active, mean_mse, std_mse, mean_infidelity,
    std_infidelity, mean_mse_masked, std_mse_masked, mean_mse_transition,
    std_mse_transition].
    """
    if approach not in APPROACH_REGISTRY:
        raise ValueError(f"approach must be one of {list(APPROACH_REGISTRY)}, got {approach!r}")
    entry = APPROACH_REGISTRY[approach]
    if entry["is_rl"] and plant_mismatch > 0:
        raise ValueError("plant_mismatch has no RL equivalent -- only valid for approach='classical'.")

    conditions = conditions or build_default_conditions()
    classical_kwargs = classical_kwargs or {}
    rl_kwargs = rl_kwargs or {}
    n = conditions.num_modulators
    csv_path = csv_path or _SWEEP_DEFAULT_CSV[approach]

    if not entry["is_rl"]:
        trials = n_mismatch_trials if plant_mismatch > 0 else 1
        fixed_policy = None
        mismatch_note = (f", {trials} plant-mismatch trial(s) each "
                          f"(plant_mismatch={plant_mismatch}, mode={plant_mismatch_mode})")
    else:
        trials = 1
        cached = None if force_retrain else _load_cached_rl(approach, conditions, rl_kwargs)
        if cached is not None:
            fixed_policy = cached
        else:
            fixed_policy = entry["policy_class"](conditions, **rl_kwargs)
            fixed_policy.train()
            _save_cached_rl(approach, fixed_policy, conditions, rl_kwargs)
        mismatch_note = " (one trained policy, reused deterministically)"

    print(f"Sweeping {approach} over all {2 ** n - 1} activation patterns "
          f"of {n} modulators{mismatch_note}...")

    records = []

    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            channels = list(combo)
            trial_values = {m: [] for m in _SWEEP_METRICS}

            for trial in range(trials):
                if not entry["is_rl"]:
                    policy = ClassicalControl(
                        conditions, plant_mismatch=plant_mismatch,
                        plant_mismatch_mode=plant_mismatch_mode, rng=trial,
                        **classical_kwargs,
                    )
                    policy.train()  # no-op
                else:
                    policy = fixed_policy

                v_mod_hist, v_src, active, target_powers = policy.solve(active_channels=channels)
                results = policy.evaluate(v_mod_hist, active, target_powers)
                for m, key in _SWEEP_METRICS.items():
                    trial_values[m].append(results[key])

            row = [k]
            for m in _SWEEP_METRICS:
                row += [float(np.mean(trial_values[m])), float(np.std(trial_values[m]))]
            records.append(row)

    records = np.array(records)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["num_active_modulators", "num_arrangements", "n_mismatch_trials"]
        for m in _SWEEP_METRICS:
            header += [f"avg_{m}", f"avg_{m}_std", f"config_{m}_std"]
        writer.writerow(header)

        for k in range(1, n + 1):
            group = records[records[:, 0] == k]
            row = [k, len(group), trials]
            col = 1
            for _ in _SWEEP_METRICS:
                mean_col, std_col = group[:, col], group[:, col + 1]
                row += [float(np.mean(mean_col)), float(np.mean(std_col)), float(np.std(mean_col))]
                col += 2
            writer.writerow(row)

    print(f"Wrote {csv_path}")
    return records



#Syntax for training the various approaches

if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    # python evaluate.py                     -> runs all four approaches
    # python evaluate.py classical            -> classical control only
    # python evaluate.py v_cap_estimate        -> v_cap_estimate approach only
    # python evaluate.py standard_rl           -> standard_rl approach only
    # python evaluate.py scaled_rl             -> scaled_rl approach only
    # python evaluate.py sweep [approach]      -> sweep over every activation
    #                                            pattern for `approach`
    #                                            (default "classical"),
    #                                            written to CSV
    approach_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if approach_arg == "sweep":
        sweep_approach = sys.argv[2] if len(sys.argv) > 2 else "classical"
        sweep_active_configurations(sweep_approach, build_default_conditions())
    else:
        conditions = build_default_conditions()
        results = eval(approach_arg, conditions, active_channels=[0, 1, 2],
                        show_baseline=True, show_raw=False)

        plt.show()