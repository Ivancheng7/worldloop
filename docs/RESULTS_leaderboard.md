# WorldLoop leaderboard

## Coverage

| model | cases | per protocol |
|---|---|---|
| `oracle_sim` | 25 | counterfactual 6, drift 1, loop360 6, null_action 6, occlusion 6 |
| `static` | 25 | counterfactual 6, drift 1, loop360 6, null_action 6, occlusion 6 |
| `wan_t2v` | 12 | counterfactual 3, loop360 3, null_action 3, occlusion 3 |
| `yume_5b` | 19 | counterfactual 6, drift 1, loop360 4, null_action 4, occlusion 4 |
| `oracle_sim_ood` | 15 | counterfactual 3, loop360 4, null_action 4, occlusion 4 |
| `static_ood` | 15 | counterfactual 3, loop360 4, null_action 4, occlusion 4 |

## Raw metrics

| model | Flow dir. cos | Flow mag. ratio | Axis purity | Switch latency (f) | Revisit DINO | Revisit SSIM | Revisit LPIPS | Permanence DINO | Drift / s | Null motion px | CF action sep | CF null drift | ASR (legacy) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `oracle_sim` | 0.998 | 0.999 | 1.000 | 1.00 | 1.000 | 0.989 | 0.006 | 0.988 | -0.006 | 0.023 | 0.263 | 0.000 | 100 |
| `static` | 0.057 | 0.000 | - | - | 1.000 | 0.989 | 0.006 | 0.988 | -0.000 | 0.023 | 0.000 | 0.000 | 0.000 |
| `wan_t2v` | 0.143 | 0.026 | 0.408 | - | 0.144 | 0.443 | 0.752 | 0.195 | 0.017 | 1.12 | 0.000 | 0.957 | 0.000 |
| `yume_5b` | 0.969 | 0.843 | 0.997 | 5.00 | 0.352 | 0.400 | 0.830 | 0.191 | 0.076 | 25.60 | 0.212 | 0.540 | 99.81 |
| `oracle_sim_ood` | 1.000 | 0.999 | 1.000 | 1.00 | 0.999 | 0.989 | 0.005 | 0.981 | -0.008 | 0.026 | 0.483 | 0.000 | 100 |
| `static_ood` | 0.135 | 0.000 | - | - | 0.999 | 0.989 | 0.005 | 0.981 | -0.000 | 0.026 | 0.000 | 0.000 | 0.000 |

## Group scores (0-100, higher is better)

| model | Control | Memory | Stability | Causality | **World Score** |
|---|---|---|---|---|---|
| `oracle_sim` | 95.75 | 99.23 | 94.07 | 100 | 97.47 |
| `static` | 5.70 | 99.23 | 99.86 | 0.000 | 23.78 |
| `wan_t2v` | 27.50 | 26.08 | 76.09 | 0.000 | 26.78 |
| `yume_5b` | 78.04 | 31.42 | 0.000 | 26.84 | 49.52 |
| `oracle_sim_ood` | 95.83 | 98.98 | 92.32 | 100 | 97.39 |
| `static_ood` | 13.46 | 98.98 | 99.84 | 0.000 | 36.49 |

## Cost (same GPU unless noted)

| model | s/chunk | realtime factor |
|---|---|---|
| `oracle_sim` | 2.28 | 1.37 |
| `static` | 0.007 | 455 |
| `wan_t2v` | 96.89 | 0.027 |
| `yume_5b` | 190 | 0.011 |
| `oracle_sim_ood` | 1.86 | 1.37 |
| `static_ood` | 0.006 | 459 |

## Per-protocol detail

| model | protocol | flow_cos | flow_mag | int_yaw_deg | revisit_ssim | revisit_dino | revisit_est_dino | permanence | drift/s | null px |
|---|---|---|---|---|---|---|---|---|---|---|
| `oracle_sim` | counterfactual | 1.000 | 1.000 | -0.081 | 0.987 | 1.000 | 0.994 | 0.992 | 0.035 | - |
| `oracle_sim` | drift | 1.000 | 0.999 | 360 | 0.987 | 0.999 | 0.990 | 0.995 | 0.001 | - |
| `oracle_sim` | loop360 | 0.994 | 0.999 | 360 | 0.990 | 1.000 | 0.990 | 0.988 | -0.011 | - |
| `oracle_sim` | null_action | - | - | -0.034 | - | - | - | - | -0.000 | 0.023 |
| `oracle_sim` | occlusion | 1.000 | 1.000 | -0.103 | 0.990 | 1.000 | 0.996 | 0.989 | -0.008 | - |
| `static` | counterfactual | 0.221 | 0.000 | -0.033 | 0.987 | 1.000 | 0.998 | 0.992 | -0.000 | - |
| `static` | drift | 0.000 | 0.000 | -0.015 | 0.987 | 0.999 | 0.999 | 0.995 | 0.000 | - |
| `static` | loop360 | 0.000 | 0.000 | -0.051 | 0.990 | 1.000 | 0.997 | 0.988 | -0.000 | - |
| `static` | null_action | - | - | -0.034 | - | - | - | - | -0.000 | 0.023 |
| `static` | occlusion | 0.171 | 0.000 | -0.051 | 0.990 | 1.000 | 0.997 | 0.989 | -0.000 | - |
| `wan_t2v` | counterfactual | 0.089 | 0.017 | -0.483 | 0.251 | 0.062 | 0.073 | 0.338 | 0.035 | - |
| `wan_t2v` | loop360 | 0.069 | 0.029 | -2.40 | 0.443 | 0.144 | 0.165 | 0.195 | 0.017 | - |
| `wan_t2v` | null_action | - | - | 0.907 | - | - | - | - | 0.055 | 1.12 |
| `wan_t2v` | occlusion | 0.216 | 0.022 | -2.40 | 0.443 | 0.144 | 0.165 | 0.195 | 0.017 | - |
| `yume_5b` | counterfactual | 0.660 | 1.02 | 119 | 0.413 | 0.428 | 0.476 | 0.285 | 0.109 | - |
| `yume_5b` | drift | 0.998 | 0.895 | 301 | 0.254 | 0.366 | 0.239 | 0.263 | 0.028 | - |
| `yume_5b` | loop360 | 0.995 | 0.931 | 382 | 0.460 | 0.250 | 0.164 | 0.169 | 0.111 | - |
| `yume_5b` | null_action | - | - | 28.77 | - | - | - | - | 0.152 | 25.60 |
| `yume_5b` | occlusion | 0.915 | 0.704 | 64.15 | 0.485 | 0.439 | 0.528 | 0.213 | 0.088 | - |
| `oracle_sim_ood` | counterfactual | 1.000 | 0.999 | -0.100 | 0.990 | 1.000 | 0.995 | 0.980 | 0.064 | - |
| `oracle_sim_ood` | loop360 | 1.000 | 0.999 | 360 | 0.989 | 0.999 | 0.991 | 0.983 | -0.005 | - |
| `oracle_sim_ood` | null_action | - | - | -0.039 | - | - | - | - | 0.000 | 0.026 |
| `oracle_sim_ood` | occlusion | 1.000 | 0.999 | -0.143 | 0.989 | 0.999 | 0.994 | 0.980 | -0.010 | - |
| `static_ood` | counterfactual | 0.236 | 0.000 | -0.074 | 0.990 | 1.000 | 0.997 | 0.980 | -0.000 | - |
| `static_ood` | loop360 | 0.000 | 0.000 | -0.059 | 0.989 | 0.999 | 0.997 | 0.983 | -0.000 | - |
| `static_ood` | null_action | - | - | -0.039 | - | - | - | - | 0.000 | 0.026 |
| `static_ood` | occlusion | 0.269 | 0.000 | -0.059 | 0.989 | 0.999 | 0.997 | 0.980 | -0.000 | - |

## Counterfactual detail (per group)

| model | group | action sep | null drift | return gap | legacy ASR |
|---|---|---|---|---|---|
| `oracle_sim` | abandoned_bakery__cf | 0.169 | 0.000 | 0.154 | 100 |
| `oracle_sim` | empty_warehouse_01__cf | 0.357 | 0.000 | 0.324 | 100 |
| `static` | abandoned_bakery__cf | 0.000 | 0.000 | 0.000 | 0.000 |
| `static` | empty_warehouse_01__cf | 0.000 | 0.000 | 0.000 | 0.000 |
| `wan_t2v` | abandoned_bakery__cf | 0.000 | 0.957 | 0.000 | 0.000 |
| `yume_5b` | abandoned_bakery__cf | 0.356 | 0.581 | 0.526 | 100 |
| `yume_5b` | empty_warehouse_01__cf | 0.068 | 0.500 | 0.190 | 99.63 |
| `oracle_sim_ood` | hotel_room__cf | 0.483 | 0.000 | 0.438 | 100 |
| `static_ood` | hotel_room__cf | 0.000 | 0.000 | 0.000 | 0.000 |

_Component definitions (all mapped to 0-100, 100 = a perfect analytic renderer):_
_Control_ = mean(flow_cos, axis_purity, max(0, 1 − switch_latency/8)). 
_Memory_ = mean(revisit_dino, revisit_ssim, permanence_dino). 
_Stability_ = mean(max(0, 1 − null_motion/8), max(0, 1 − |drift/s|/0.05)). 
_Causality_ = clip01(mean CF action sep / 0.25) × 1/(1 + mean CF null drift / 0.25), both means taken over counterfactual groups: the action branches must separate from the no-op branch *and* the no-op branch must hold still. `ASR (legacy)` = post/pre with a ×100 cap; it is kept for audit only and is no longer scored, because `pre` is legitimately 0 for a deterministic model, which saturates the ratio for every model that renders a stable prefix. `-` means the counterfactual protocol was not run for that model. 
_**World Score = sqrt(Control × Memory)** — a world model must both move when told and keep the world it moved through. This is deliberately a geometric mean: doing nothing scores Control 0, and regenerating a new world every chunk scores Memory ~0. 
_The raw metrics above are the primary record; group scores exist to expose the trade-off (a static model wins Memory and loses Control)._
