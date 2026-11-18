# Open world models, November 2026 — what actually runs on one 16 GB GPU

Compiled while choosing the subjects for WorldLoop. Every row was checked against
the upstream README / model card; "measured" numbers come from this machine
(RTX 5060 Ti 16 GB) and are marked as such.

| model | org · date | params | VRAM requirement | interface | licence | 16 GB verdict |
|---|---|---|---|---|---|---|
| **Yume-5B (Yume-1.5)** | stdstu12 / SJTU-PKU group · 2025-12 | 5 B | official: runs on a 16 GB laptop 4090 | image + text + keyboard/mouse tokens | Apache-2.0 | ✅ **used**: one process, 33-frame chunks, ~2 s of video per chunk |
| **Matrix-Game 3.5** | Riemann Dynamics · 2026-08 | 5 B (Wan2.2-TI2V) | ≥ 40 GB at 704×1280 (authors) | image + **camera trajectory (.npz)** + text | Apache-2.0 | ❌ locally, ✅ via the official HF Space |
| **Matrix-Game 3.0** | Skywork · 2026-03 | 5 B | 24 GB + FlashAttention-3, A/H-class tested | image + text + action | Apache-2.0 | ❌ |
| **Matrix-Game 4.0** | Skywork · 2026-10 | not published | 9.2 GB measured by authors on RTX 5070 Ti (16 GB) preset | **low-poly engine video** + text → neural rendering | Apache-2.0 | ⚠️ Linux + CUDA 13.2 only; different paradigm (engine holds physics) |
| **ABot-World-0** | AMap CV Lab · 2026-07 | 5 B | 19 GB measured on RTX 5090 | image + action, infinite rollout | Apache-2.0 | ❌ (3 GB over) + heaviest install chain |
| **HY-WorldPlay** | Tencent Hunyuan · 2025-12 | 8 B | 28 GB (sp=8) … 72 GB (sp=1) | image + keyboard poses, streaming | Tencent community | ❌ |
| **LingBot-World** | Robbyant (Ant) · 2026-01 | A14B (42 B total) | 8× GPU FSDP | image + camera poses / action strings | Apache-2.0 | ❌ |
| **Cosmos 3 Edge** | NVIDIA · 2026-07 | 4 B | positioned for Jetson Thor / RTX Pro 6000 | text/image/audio/action → video/action | OpenMDW-1.1 | ⚠️ untested, gated guardrail model |
| **Cosmos 3 Nano** | NVIDIA · 2026-05 | 16 B | RTX Pro 6000 class | omnimodal | OpenMDW-1.1 | ❌ |
| **Oasis-500M** | Etched × Decart · 2024-11 | 0.5 B | < 8 GB | Minecraft frames + keyboard/mouse | MIT-ish (check repo) | ✅ compute-wise, ❌ domain-wise: trained on Minecraft, a photoreal frame is out of distribution |
| **V-JEPA 2** | Meta · 2025-06 | 0.3–1 B | small | video → latent (no generation) | MIT | ✅ but not a video world model |
| **Genie 3 / Project Genie** | DeepMind · 2025-08 | — | — | — | closed | ❌ no weights |

## Why this matters for evaluation design

The runnable set on a consumer GPU is essentially **one model** (Yume-5B), plus a
pose-conditioned model reachable through a hosted demo (Matrix-Game 3.5). Everything
else is either an 8-GPU pipeline or over the memory budget. A benchmark that only
reports "we evaluated the newest models" is therefore usually reporting numbers
produced by someone with A100s — which is exactly why WorldLoop is designed to run
in ~1.5 GB of metric memory and to accept *any* `step(action) -> frames` backend,
local or remote.

## Interface taxonomy

Three incompatible action interfaces exist in the wild, and the benchmark's adapter
layer has to hide all three:

1. **Discrete keyboard tokens** (Yume, Matrix-Game 1/2/3, Oasis, ABot): action is a
   string or an integer id per chunk. Gain is unknown and must be measured.
2. **Camera poses** (Matrix-Game 3.5, LingBot-World, HY-WorldPlay): action is a
   6-DoF trajectory, usually `c2w` matrices plus intrinsics. Gain is explicit, and
   this is the only interface where analytic ground-truth flow can be matched
   exactly.
3. **Latent / visual actions** (research code): a learned action embedding; the
   semantics are only defined up to the training data.

WorldLoop v0.1 defines its own chunk-level language (`turn_left/turn_right/
pitch_up/pitch_down/noop` at 60°/s and 30°/s) and lets each adapter compile it; the
adapter records the mapping in `meta.json` so that gain mismatches are visible
rather than silently averaged away.
