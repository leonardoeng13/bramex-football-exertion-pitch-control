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

## First model check

The first leakage-safe models use current possession, ball location, past-only ball displacement/speed, recent switch counts, recent live-ball share and time within the half. Hyperparameters were fixed before leave-one-match-out evaluation.

| Candidate horizon | Historical mean | Poisson | Gradient boosting |
|---:|---:|---:|---:|
| 30 s | **1.427** | 1.441 | 1.447 |
| 60 s | **2.030** | 2.036 | 2.115 |
| 300 s | **4.931** | 5.053 | 5.517 |

The simple feature set does not beat the historical-mean baseline. Therefore, there is currently no evidence for a predictive improvement. This is a useful negative control, not a final rejection of the workstream hypothesis: the shared pitch-control, pressure, player-load and game-state features are not yet included.

## Decisions required from the team

1. Official forecast horizon and forecast cadence.
2. Final submission schema and whether forecasts are required during dead-ball periods.
3. Shared definition and implementation of pitch control.
4. Shared player-load and pressure features available at each forecast origin.
5. Whether shot/foul stretch targets remain in scope before the required targets beat baseline.

## Next analysis step

After the shared pipeline is approved, add pitch-control change, local control near the ball, team compactness/pressure, possession duration, score state and player-load summaries. Re-run the same frozen outer validation and retain the historical-mean result as the required baseline.
