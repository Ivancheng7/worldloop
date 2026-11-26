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
| `yume_5b_ood` | 15 | counterfactual 3, loop360 4, null_action 4, occlusion 4 |

## Raw metrics

| model | Flow dir. cos | Flow mag. ratio | Axis purity | Switch latency (f) | Revisit DINO | Revisit DINO floor | Revisit DINO (floor-rel.) | Revisit SSIM | Revisit SSIM floor | Revisit SSIM (floor-rel.) | Revisit LPIPS | Permanence DINO | Permanence floor | Permanence (floor-rel.) | Drift / s | Null motion px | CF action sep | CF null drift | ASR (legacy) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `oracle_sim` | 0.998 | 0.999 | 1.000 | 1.00 | 1.000 | 0.572 | 0.999 | 0.989 | 0.345 | 0.982 | 0.006 | 0.988 | 0.359 | 0.980 | -0.006 | 0.023 | 0.263 | 0.000 | 100 |
| `static` | 0.057 | 0.000 | - | - | 1.000 | 0.593 | 0.998 | 0.989 | 0.357 | 0.982 | 0.006 | 0.988 | 0.359 | 0.980 | -0.000 | 0.023 | 0.000 | 0.000 | 0.000 |
| `wan_t2v` | 0.143 | 0.026 | 0.408 | - | 0.144 | 0.203 | 0.029 | 0.443 | 0.403 | 0.070 | 0.752 | 0.195 | 0.203 | 0.017 | 0.017 | 1.12 | 0.000 | 0.957 | 0.000 |
| `yume_5b` | 0.969 | 0.843 | 0.997 | 5.00 | 0.352 | 0.350 | 0.066 | 0.400 | 0.363 | 0.070 | 0.830 | 0.191 | 0.190 | 0.018 | 0.076 | 25.60 | 0.212 | 0.540 | 99.81 |
| `oracle_sim_ood` | 1.000 | 0.999 | 1.000 | 1.00 | 0.999 | 0.432 | 0.999 | 0.989 | 0.532 | 0.974 | 0.005 | 0.981 | 0.364 | 0.967 | -0.008 | 0.026 | 0.483 | 0.000 | 100 |
| `static_ood` | 0.135 | 0.000 | - | - | 0.999 | 0.482 | 0.999 | 0.989 | 0.528 | 0.974 | 0.005 | 0.981 | 0.364 | 0.967 | -0.000 | 0.026 | 0.000 | 0.000 | 0.000 |
| `yume_5b_ood` | 0.856 | 0.789 | 0.997 | 4.50 | 0.317 | 0.374 | 0.093 | 0.622 | 0.572 | 0.103 | 0.684 | 0.223 | 0.232 | 0.003 | 0.120 | 15.94 | 0.322 | 0.207 | 100 |

## Group scores (0-100, higher is better)

| model | Control | Memory | Stability | Causality | **World Score** |
|---|---|---|---|---|---|
| `oracle_sim` | 95.75 | 98.70 | 94.07 | 100 | 97.22 |
| `static` | 5.70 | 98.69 | 99.86 | 0.000 | 23.71 |
| `wan_t2v` | 27.50 | 3.86 | 76.09 | 0.000 | 10.30 |
| `yume_5b` | 78.04 | 5.14 | 0.000 | 26.84 | 20.04 |
| `oracle_sim_ood` | 95.83 | 98.01 | 92.32 | 100 | 96.91 |
| `static_ood` | 13.46 | 98.01 | 99.84 | 0.000 | 36.32 |
| `yume_5b_ood` | 76.36 | 6.62 | 0.000 | 54.74 | 22.49 |

## Cost (same GPU unless noted)

| model | s/chunk | realtime factor |
|---|---|---|
| `oracle_sim` | 2.28 | 1.37 |
| `static` | 0.007 | 455 |
| `wan_t2v` | 96.89 | 0.027 |
| `yume_5b` | 190 | 0.011 |
| `oracle_sim_ood` | 1.86 | 1.37 |
| `static_ood` | 0.006 | 459 |
| `yume_5b_ood` | 197 | 0.011 |

## Per-protocol detail

| model | protocol | flow_cos | flow_mag | int_yaw_deg | revisit_ssim | revisit_ssim_norm | revisit_dino | revisit_dino_norm | permanence | permanence_norm | drift/s | null px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `oracle_sim` | counterfactual | 1.000 | 1.000 | -0.081 | 0.987 | 0.982 | 1.000 | 0.998 | 0.992 | 0.982 | 0.035 | - |
| `oracle_sim` | drift | 1.000 | 0.999 | 360 | 0.987 | 0.984 | 0.999 | 0.998 | 0.995 | 0.981 | 0.001 | - |
| `oracle_sim` | loop360 | 0.994 | 0.999 | 360 | 0.990 | 0.982 | 1.000 | 0.999 | 0.988 | 0.980 | -0.011 | - |
| `oracle_sim` | null_action | - | - | -0.034 | - | - | - | - | - | - | -0.000 | 0.023 |
| `oracle_sim` | occlusion | 1.000 | 1.000 | -0.103 | 0.990 | 0.981 | 1.000 | 0.999 | 0.989 | 0.980 | -0.008 | - |
| `static` | counterfactual | 0.221 | 0.000 | -0.033 | 0.987 | 0.982 | 1.000 | 0.998 | 0.992 | 0.982 | -0.000 | - |
| `static` | drift | 0.000 | 0.000 | -0.015 | 0.987 | 0.984 | 0.999 | 0.997 | 0.995 | 0.981 | 0.000 | - |
| `static` | loop360 | 0.000 | 0.000 | -0.051 | 0.990 | 0.981 | 1.000 | 0.999 | 0.988 | 0.980 | -0.000 | - |
| `static` | null_action | - | - | -0.034 | - | - | - | - | - | - | -0.000 | 0.023 |
| `static` | occlusion | 0.171 | 0.000 | -0.051 | 0.990 | 0.981 | 1.000 | 0.999 | 0.989 | 0.980 | -0.000 | - |
| `wan_t2v` | counterfactual | 0.089 | 0.017 | -0.483 | 0.251 | 0.008 | 0.062 | 0.027 | 0.338 | 0.000 | 0.035 | - |
| `wan_t2v` | loop360 | 0.069 | 0.029 | -2.40 | 0.443 | 0.010 | 0.144 | 0.043 | 0.195 | 0.017 | 0.017 | - |
| `wan_t2v` | null_action | - | - | 0.907 | - | - | - | - | - | - | 0.055 | 1.12 |
| `wan_t2v` | occlusion | 0.216 | 0.022 | -2.40 | 0.443 | 0.130 | 0.144 | 0.014 | 0.195 | 0.017 | 0.017 | - |
| `yume_5b` | counterfactual | 0.660 | 1.02 | 119 | 0.413 | 0.077 | 0.428 | 0.161 | 0.285 | 0.026 | 0.109 | - |
| `yume_5b` | drift | 0.998 | 0.895 | 301 | 0.254 | 0.000 | 0.366 | 0.050 | 0.263 | 0.000 | 0.028 | - |
| `yume_5b` | loop360 | 0.995 | 0.931 | 382 | 0.460 | 0.020 | 0.250 | 0.036 | 0.169 | 0.014 | 0.111 | - |
| `yume_5b` | null_action | - | - | 28.77 | - | - | - | - | - | - | 0.152 | 25.60 |
| `yume_5b` | occlusion | 0.915 | 0.704 | 64.15 | 0.485 | 0.189 | 0.439 | 0.112 | 0.213 | 0.023 | 0.088 | - |
| `oracle_sim_ood` | counterfactual | 1.000 | 0.999 | -0.100 | 0.990 | 0.978 | 1.000 | 0.999 | 0.980 | 0.971 | 0.064 | - |
| `oracle_sim_ood` | loop360 | 1.000 | 0.999 | 360 | 0.989 | 0.974 | 0.999 | 0.999 | 0.983 | 0.970 | -0.005 | - |
| `oracle_sim_ood` | null_action | - | - | -0.039 | - | - | - | - | - | - | 0.000 | 0.026 |
| `oracle_sim_ood` | occlusion | 1.000 | 0.999 | -0.143 | 0.989 | 0.974 | 0.999 | 0.999 | 0.980 | 0.964 | -0.010 | - |
| `static_ood` | counterfactual | 0.236 | 0.000 | -0.074 | 0.990 | 0.978 | 1.000 | 0.999 | 0.980 | 0.971 | -0.000 | - |
| `static_ood` | loop360 | 0.000 | 0.000 | -0.059 | 0.989 | 0.974 | 0.999 | 0.999 | 0.983 | 0.970 | -0.000 | - |
| `static_ood` | null_action | - | - | -0.039 | - | - | - | - | - | - | 0.000 | 0.026 |
| `static_ood` | occlusion | 0.269 | 0.000 | -0.059 | 0.989 | 0.974 | 0.999 | 0.999 | 0.980 | 0.964 | -0.000 | - |
| `yume_5b_ood` | counterfactual | 0.744 | 0.745 | 33.52 | 0.695 | 0.039 | 0.336 | 0.226 | 0.291 | 0.043 | 0.126 | - |
| `yume_5b_ood` | loop360 | 0.994 | 0.831 | 333 | 0.618 | 0.070 | 0.258 | 0.062 | 0.208 | 0.006 | 0.128 | - |
| `yume_5b_ood` | null_action | - | - | 13.50 | - | - | - | - | - | - | 0.041 | 15.94 |
| `yume_5b_ood` | occlusion | 0.718 | 0.747 | 79.92 | 0.625 | 0.136 | 0.376 | 0.124 | 0.238 | 0.000 | 0.112 | - |

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
| `yume_5b_ood` | hotel_room__cf | 0.322 | 0.207 | 0.645 | 100 |

_Component definitions (all mapped to 0-100, 100 = a perfect analytic renderer):_
_Control_ = mean(flow_cos, axis_purity, max(0, 1 − switch_latency/8)). 
_Memory_ = mean(revisit_dino_norm, revisit_ssim_norm, permanence_dino_norm), each a similarity scored *relative to a negative-pair floor*: the same DINO/SSIM instrument applied to pairs that are definitely not revisits (the reference against frames with the commanded camera ≥120° away; for permanence, probe A's patch against probe B's direction). Similarity without that subtraction measures image smoothness as much as memory — Yume's revisit DINO was 0.352 indoors against its own 0.350 floor, i.e. no more like the start view than like a random other direction, and its out-of-distribution SSIM of 0.622 was tracking a 0.572 floor. Raw values are kept in the per-protocol table for audit. 
_Stability_ = mean(max(0, 1 − null_motion/8), max(0, 1 − |drift/s|/0.05)). 
_Causality_ = clip01(mean CF action sep / 0.25) × 1/(1 + mean CF null drift / 0.25), both means taken over counterfactual groups: the action branches must separate from the no-op branch *and* the no-op branch must hold still. `ASR (legacy)` = post/pre with a ×100 cap; it is kept for audit only and is no longer scored, because `pre` is legitimately 0 for a deterministic model, which saturates the ratio for every model that renders a stable prefix. `-` means the counterfactual protocol was not run for that model. 
_**World Score = sqrt(Control × Memory)** — a world model must both move when told and keep the world it moved through. This is deliberately a geometric mean: doing nothing scores Control 0, and regenerating a new world every chunk scores Memory ~0. 
_The raw metrics above are the primary record; group scores exist to expose the trade-off (a static model wins Memory and loses Control)._
