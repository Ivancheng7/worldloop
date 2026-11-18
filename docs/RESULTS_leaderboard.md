# WorldLoop leaderboard

| model | Flow dir. cos | Flow mag. ratio | Axis purity | Switch latency (f) | Revisit DINO | Revisit SSIM | Revisit LPIPS | Permanence DINO | Drift / s | Null motion px | ASR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `oracle_sim` | 0.998 | 0.999 | 1.000 | 1.00 | 1.000 | 0.989 | 0.006 | 0.988 | -0.006 | 0.023 | 100 |
| `static` | 0.057 | 0.000 | - | - | 1.000 | 0.989 | 0.006 | 0.988 | -0.000 | 0.023 | 0.000 |
| `wan_t2v` | 0.143 | 0.026 | 0.408 | - | 0.144 | 0.443 | 0.752 | 0.195 | 0.017 | 1.37 | 0.000 |
| `yume_5b` | 0.631 | 0.690 | 0.997 | 4.00 | 0.331 | 0.460 | 0.766 | 0.254 | 0.073 | 11.78 | 8.33 |

## Group scores (0-100, higher is better)

| model | Control | Memory | Stability | Causality | **World Score** |
|---|---|---|---|---|---|
| `oracle_sim` | 95.75 | 99.23 | 94.07 | 97.09 | 97.47 |
| `static` | 5.70 | 99.23 | 99.86 | 0.000 | 23.78 |
| `wan_t2v` | 27.50 | 26.08 | 74.52 | 0.001 | 26.78 |
| `yume_5b` | 70.93 | 34.85 | 0.000 | 73.53 | 49.72 |

## Cost (same GPU unless noted)

| model | s/chunk | realtime factor |
|---|---|---|
| `oracle_sim` | 2.28 | 1.37 |
| `static` | 0.007 | 417 |
| `wan_t2v` | 91.51 | 0.027 |
| `yume_5b` | 691 | 0.008 |

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
| `wan_t2v` | null_action | - | - | 1.70 | - | - | - | - | 0.065 | 1.37 |
| `wan_t2v` | occlusion | 0.216 | 0.022 | -2.40 | 0.443 | 0.144 | 0.165 | 0.195 | 0.017 | - |
| `yume_5b` | counterfactual | 0.055 | 0.465 | 63.49 | 0.273 | 0.406 | 0.353 | 0.449 | 0.061 | - |
| `yume_5b` | loop360 | 0.764 | 0.648 | 204 | 0.471 | 0.330 | 0.361 | 0.244 | 0.074 | - |
| `yume_5b` | null_action | - | - | 0.052 | - | - | - | - | -0.019 | 11.78 |
| `yume_5b` | occlusion | 0.497 | 0.733 | 183 | 0.449 | 0.333 | 0.445 | 0.263 | 0.072 | - |

_Component definitions (all mapped to 0-100, 100 = a perfect analytic renderer):_
_Control_ = mean(flow_cos, axis_purity, max(0, 1 − switch_latency/8)). 
_Memory_ = mean(revisit_dino, revisit_ssim, permanence_dino). 
_Stability_ = mean(max(0, 1 − null_motion/8), max(0, 1 − |drift/s|/0.05)). 
_Causality_ = ASR/(ASR+3); `-` means the counterfactual protocol was not run for that model. 
_**World Score = sqrt(Control × Memory)** — a world model must both move when told and keep the world it moved through. This is deliberately a geometric mean: doing nothing scores Control 0, and regenerating a new world every chunk scores Memory ~0. 
_The raw metrics above are the primary record; group scores exist to expose the trade-off (a static model wins Memory and loses Control)._
