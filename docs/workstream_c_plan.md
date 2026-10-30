# Workstream C - possession changes and chances

> Draft for team review. This is not a final result and should not be presented externally before team approval.

## Course-aligned objective

The required Workstream C prediction target is the number of Home/Away possession switches in a future window, counting only switches during uninterrupted active play. The official score is mean absolute error (MAE). Shot and foul occurrence are stretch targets evaluated with average precision. The course has not yet specified the final forecast horizon or submission schema in the available materials.

## Operational definition

A possession switch is a Home-to-Away or Away-to-Home change between consecutive 25 Hz frames where the ball is alive in both frames. A change across a dead-ball interval is excluded. Forecast origins are restricted to frames where the ball is alive, and every feature uses only the origin frame or earlier frames.

## Data audit

- Six complete training matches, each with match information, events and observed positions.
- Twelve halves at 25 Hz on a 105 x 68 m pitch.
- No events occur after the corresponding tracking interval.
- Active-play possession switches range from 278 to 408 per match.
- The `floodlight` gameclock warnings are caused by normal stoppage time beyond its generic 45-minute validation range, not by event/tracking misalignment.

## Pre-registered analysis

### Hypothesis

Loss of local/team pitch control, backward ball movement and recent pressure precede possession switches. Features available before a forecast origin may reduce out-of-match MAE relative to a historical-rate baseline.

### Unit and controls

The unit is a rolling forecast origin within match and half during active play. Planned controls include match, Home/Away side, half, score/game state, pitch zone, current possession duration and time in match.

### Validation

- Outer leave-one-match-out validation across all six training matches.
- Any later hyperparameter tuning must use match-grouped folds inside the five training matches only.
- Report MAE for every held-out match, the across-match mean, matches beating baseline and uncertainty over matches.
- Refit and freeze only after the target horizon, features and model have been approved.

## Provisional horizon sensitivity

These horizons test the pipeline; they do not select the final course horizon.

| Candidate horizon | Mean future switches | Zero-target share | Historical-mean LOMO MAE |
|---:|---:|---:|---:|
| 30 s | 2.024 | 21.9% | 1.427 |
| 60 s | 3.723 | 6.2% | 2.030 |
| 300 s | 17.156 | 0.0% | 4.931 |

## Shared-pipeline model check and Workstream B integration

Targets are counted from the original 25 Hz possession flags. Predictors are aligned to the team's shared 5 Hz tables and include current and past-only ball position, Voronoi control, ball-zone control, HSR, distance, possession duration, recent switches, live-ball share and recent events. Hyperparameters were fixed before leave-one-match-out evaluation.

Monica's Workstream B notebook identifies possession share, pitch share, ball-zone control, progression and control at the start of a possession as the main team-context signals. Its published integration rows summarize the complete current minute, which is not safe for a forecast made inside that minute. Workstream C therefore recomputes those concepts at each 5-second origin using only the origin and earlier samples. The local shared cache contains the agreed Voronoi alternative; the implementation automatically uses TTI when those columns are available.

| Horizon | Historical mean | C-only Poisson | C+B Poisson | C-only HGB | C+B HGB |
|---:|---:|---:|---:|---:|---:|
| 30 s | 1.427 | 1.431 | **1.427** | 1.454 | 1.448 |
| 60 s | 2.030 | **2.021** | 2.033 | 2.050 | 2.081 |
| 300 s | **4.931** | 5.237 | 5.287 | 5.548 | 5.605 |

At 60 seconds, the C-only Poisson model improves mean MAE by 0.009 and beats the baseline in four of six held-out matches. The paired match-bootstrap interval for model-minus-baseline MAE is -0.028 to 0.015, so the result does not establish a reliable predictive improvement. Adding the leakage-safe B signals changes the 60-second Poisson delta to +0.002 (95% match-bootstrap interval -0.016 to 0.023; three of six wins): useful team context, but no incremental forecasting gain. At 30 seconds C+B is effectively tied with the baseline, while both feature sets are worse at 300 seconds. No model is frozen from these provisional horizons.

## Decisions required from the team

1. Official forecast horizon and forecast cadence.
2. Final submission schema and whether forecasts are required during dead-ball periods.
3. Team approval of the shared Voronoi definition and whether time-to-arrival control is required.
4. Whether the team will share/build the TTI cache for a final sensitivity run; the current C integration uses the Voronoi alternative.
5. Whether to retain the B variables as context despite no incremental MAE gain, and whether shot/foul stretch targets remain in scope.

## Next analysis step

The Workstream B ablation and the frozen outer validation are complete. Do not open the holdout or select a final model until the team confirms the official horizon, forecast cadence, submission schema and pitch-control definition. If a shared TTI cache becomes available, repeat the same locked ablation without changing models or thresholds.
