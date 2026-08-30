# Classical Control vs. RL Shared API

The files in this folder allow comparison of the classical control approach and three reinforcement learning approaches to predistorting optical pulses.
From evaluate.py, the user can input environmental conditions and then output the MSE/infidelity values for any one, or all four, strategies.

## Architecture

Eight files, four layers:

```
physics.py            <- defines the conditions and logic of the simulated circuit environment
      |
policy.py              <- shared base class for all four approaches defining the construction of the environment and target,
      |                   smoothing, scoring, plotting
      |
  ------------------------------------------------
  |              |                |               |
classical_control.py  rl_common.py -- shared physics kernel + training
  |              |    callbacks used by all three RL approaches below
  |              |                |               |
  |         rl_v_cap_estimate.py  rl_standard.py  rl_scaled.py   <- these three files
  |              |                |               |               define the RL strategies
  ------------------------------------------------
      |
evaluate.py            <- outputs MSE/infidelity values from each approach
```

| File | Role |
|---|---|
| `physics.py` | `ModulatorArray` — the circuit dynamics (bias-tee, LC crosstalk) and electro-optic response (`calc_power`, `calc_power_inv`, `calc_infidelity`). None of the four approaches subclass this; all of them just call it directly. |
| `policy.py` | `Policy` abstract base class + `EnvironmentalConditions` dataclass. Owns everything identical across approaches: building the active-channel mask, building the target trajectory (smoothed + unsmoothed), scoring (MSE/masked-MSE/transition-MSE/infidelity), and the shared three-panel plot. Declares `train()`/`solve()` as abstract — each subclass implements its own. |
| `classical_control.py` | `ClassicalControl(Policy)`. Exact feedforward inversion of the plant's own equations — `train()` is a no-op, `solve()` is a near-instant batch computation. Also has a classical-only `plot_with_baseline()` (four panels: input voltage, voltage at load, power vs. target, residual — target vs. non-predistorted vs. predistorted), since RL has no equivalent "uncompensated" baseline. |
| `rl_common.py` | Shared infrastructure for all three RL approaches below: the physics simulation kernel and the training callbacks (`TrainingMetricsCallback`, `ProgressBarCallback`, `BestMSECallback`), which were identical across all three approaches' original standalone training scripts. **Note:** this file's physics models `cap_p` as coupling the *bias-tee* stage, not the LC stage the way `physics.py` does — a genuine, pre-existing difference in how these three RL approaches were trained versus how classical control models crosstalk. It doesn't affect scoring (which only depends on electro-optic parameters, never on the RLC values), but the same `cap_p` value does *not* imply the same physical coupling assumption across approaches. |
| `rl_v_cap_estimate.py` | `RLVCapEstimate(Policy)`. The source voltage is set to an analytically-derived hint, `v_target(t) = target_input(t) + v_cap_estimate(t)`, for the entire episode, except two short windows immediately after each pulse starts and ends, where the network may add a learned correction — larger for channels that are actively firing, much smaller for channels only exposed to crosstalk from a firing neighbor. |
| `rl_standard.py` | `RLStandard(Policy)`. A fully free, unconstrained action space (any voltage in `[action_min_v, action_max_v]`, no phase-dependent structure at all) paired with a "measured, not hinted" observation space — elapsed time, target power, the channel's own previously measured output power, its own and its neighbors' previously applied voltage, and simple timing flags. No `v_cap_estimate` anywhere. |
| `rl_scaled.py` | `RLScaled(Policy)`. The same observation space as `rl_standard.py`, but a multi-branch, phase-dependent action space: separate absolute voltage ranges for the pre-pulse/on-pulse/off-pulse/never-fires phases, with a strictly-increasing accumulator enforced during the on-pulse ramp. |
| `evaluate.py` | Configuration constants, the `eval()` function, the approach registry, the RL model cache, and `sweep_active_configurations()` for grouped statistics across every possible activation pattern. |

All three RL approaches train a `RecurrentPPO` policy (`sb3_contrib`) with `stable_baselines3` against their own private `gymnasium.Env`; `train()` actually trains, `solve()` runs the trained policy deterministically against a fixed activation pattern.

## Installation

```bash
pip install numpy scipy numba gymnasium matplotlib sb3_contrib stable_baselines3 tqdm
```

## Quick start

Edit the configuration block at the top of `evaluate.py`:

```python
NUM_MODULATORS = 5
NUM_PULSES = 1
PULSE_WIDTH = 150.0          # ns
INTER_PULSE_GAP = 1000.0     # ns
PRE_PAD = 25.0               # ns
POST_PAD = 25.0              # ns
NUM_POINTS = 2000
RISE_TIME_NS = 0
TRANSITION_WINDOW_NS = 10.0
H = 0.1
...
```

Then run:

```bash
python evaluate.py                # runs all four approaches (default)
python evaluate.py classical      # classical control only
python evaluate.py v_cap_estimate # the v_cap_estimate hint approach only
python evaluate.py standard_rl    # the free-action approach only
python evaluate.py scaled_rl      # the scaled/constrained-action approach only
python evaluate.py sweep [approach]  # sweep every activation pattern for
                                      # `approach` (default "classical"),
                                      # writing grouped statistics to CSV
```

The `active_channels` pattern to test against is currently set directly in
`evaluate.py`'s `__main__` block — edit the `active_channels=[0, 1, 2]`
argument there to test a different firing pattern (e.g. `[0, 1, 2, 3, 4]` for "all
five fire", or `None` for "all channels fire").

**Note:** `evaluate.py`'s `__main__` block calls `matplotlib.use("TkAgg")`
before any plotting happens, so the paginated figures render in an
interactive window (with zoom/pan) rather than as static inline images.
If `TkAgg` isn't available in your environment, swap it for another
interactive backend (e.g. `"QtAgg"`).

## `eval()` reference

```python
eval(approach, conditions=None, active_channels=None,
     rl_kwargs=None, classical_kwargs=None,
     force_retrain=False, plot=True, show_baseline=False, show_raw=True)
```

- **`approach`**: `"classical"`, `"v_cap_estimate"`, `"standard_rl"`,
  `"scaled_rl"`, or `"all"`. `"all"` returns a dict keyed by approach name
  instead of a flat results dict.
- **`conditions`**: an `EnvironmentalConditions` override. If omitted, built
  from the constants at the top of `evaluate.py`.
- **`active_channels`**: which modulators fire, e.g. `[0, 1, 2]`. `None`
  means every channel fires.
- **`rl_kwargs`** / **`classical_kwargs`**: passed straight through to
  whichever approach's constructor is selected (see below) — `rl_kwargs`
  is ignored for `"classical"`, `classical_kwargs` is ignored for every RL
  approach.
- **`force_retrain`**: don't use a previously trained RL model for the
  same approach + environmental parameters + hyperparameters; train from
  scratch instead.
- **`plot`**: whether to render plots.
- **`show_baseline`**: classical-only — renders the four-panel
  target/non-predistorted/predistorted/residual comparison instead of the
  shared three-panel plot.
- **`show_raw`**: classical-only, ignored unless `show_baseline` is also
  `True` — if `False`, drops the non-predistorted baseline trace/legend
  entry from that comparison plot, leaving just target vs. predistorted.

Returns the dict produced by `Policy.evaluate()`: `mse_per_channel`,
`mse_overall`, `mse_masked_per_channel`, `mse_masked_overall`,
`mse_transition_per_channel`, `mse_transition_overall`,
`infidelity_per_channel`, `mean_infidelity_overall`.

## Important: tracked target vs. scored target

All four approaches track the rise-time-smoothed
target during training/solving (RL's reward and observations; classical's
exact inversion), but the final MSE/infidelity values are scored against the
unsmoothed, idealized square-wave target.

This is defined in `Policy.build_targets()`, which returns
`(target_powers_smoothed, target_input_smoothed, target_powers_unsmoothed,
target_input_unsmoothed)`. To change which target is
tracked or scored, adjust which variant is passed through in that section.

## Masked and transition MSE

`Policy.evaluate()` also reports two additional MSE variants, split around
`conditions.transition_window_ns` (10 ns by default): for each channel,
**transition MSE** is the mean squared error computed *only* within that
window immediately following every rising and falling edge of that
channel's active mask, and **masked MSE** is the mean squared error over
everything *outside* those windows. A channel that never fires has no
transitions, so its masked MSE equals its full-episode MSE and its
transition MSE is `NaN` (excluded from the overall transition-MSE
average, rather than treated as zero).

**Important:** this windowing uses `conditions.h` directly (not a value
derived from the actual time array), so `h` must be set to match
`total_time / (num_points - 1)` for whatever `pre_pad`/`pulse_width`/
`post_pad`/`num_pulses`/`inter_pulse_gap` you're using, or the window
boundaries will silently be wrong.

### Model caching

The first `eval(approach, conditions, ...)` call for a given RL approach
and configuration trains from scratch and saves under
`./trained_models/`, keyed by approach name, that approach's own
`POLICY_VERSION`, and a hash of `conditions` + `rl_kwargs`. Every
subsequent call with the same approach, conditions, and `rl_kwargs` reuses
the cached model instead of retraining, signaled with the message:

```
Reusing cached standard_rl model at ./trained_models/standard_rl_v1_9a0a2a065f2734dd (trained ...)
```

Pass `force_retrain=True` to bypass the cache and train fresh regardless.
Classical control has no training phase, so it's never cached.

### Sweeping every activation pattern

`sweep_active_configurations(approach, conditions=None, ...)` runs a given
approach over every possible non-empty arrangement of active modulators
(`2**n - 1` arrangements for `n` modulators), groups the results by how
many modulators were active, and writes average/standard-deviation MSE,
infidelity, masked MSE, and transition MSE per group to a CSV (default
filename depends on the approach, e.g. `modulator_evaluation_classic.csv`).
For classical control, `plant_mismatch > 0` re-draws the plant across
`n_mismatch_trials` independent realizations per arrangement; for RL
approaches, `plant_mismatch` has no equivalent and raises `ValueError` if
set above `0`, since one trained policy is simply reused deterministically
across every arrangement rather than retrained per arrangement.
