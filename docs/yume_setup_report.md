# Yume-5B (Yume-1.5) 在 RTX 5060 Ti 16GB / Windows 上的落地报告

- 机器：Windows 10.0.26200，RTX 5060 Ti 16GB（sm_120 / Blackwell，capability (12,0)），驱动 610.88，
  Python 3.10.11，物理内存 31.6GB，Clash Verge（`verge-mihomo`，127.0.0.1:7897）在跑但没有代理环境变量
- 时间：2026-11-13
- 状态：**环境与代码链路全部打通并实测；33GB 权重因网络（HF 单连接 ~1.8MB/s、TLS 反复 EOF、xet 会假死）
  在 2 小时窗口内只下到 6.6GB，真推理（DiT）没跑成 —— 权重下载已由主进程接管。**
  下文所有数字标注了「实测」或「估算」，没有编的。

---

## 0. TL;DR

| 项 | 结论 |
|---|---|
| 环境能不能跑 | **能**。torch 2.11.0+cu128 在 sm_120 上 bf16 matmul 正常；整个 `wan23` + `webapp_single_gpu` 在 **不装 flash-attn / triton / apex / bitsandbytes** 的情况下正常 import 并执行 |
| 动作接口 | **不是张量**。离散键盘 token（W/A/S/D/组合）+ 鼠标 token（↑↓←→·）**拼进文本 prompt**；另有 3 个数值参数也是纯文本 |
| 帧数 | 每次（chunk）生成 **4n+1 帧**，官方/实测有效值 **33 帧**（16fps ≈ 2.06s）。总数靠循环续帧堆，理论无上限 |
| 闭环 | **官方支持**。`webapp_single_gpu.long_generate` 用 `LAST` 缓存 13 个 latent（=49 帧上下文）做 `continue_from_last`；我把状态落盘成 `.npz` 后可跨进程续 |
| 实测显存（已有） | VAE 权重 2.63GB；**encode 16 帧峰值 9.86GB，encode 33 帧峰值 10.03GB**；**decode 9 latent（33 帧）在 16GB 上 5 分钟跑不完（WDDM 溢出到共享显存），必须开 tiled decode** |
| 单 chunk 墙钟 | **未实测**（DiT 权重没到）。按 FLOPs 估算：首 chunk 30~60s，续帧 chunk 40~90s，96 帧（3 chunk）约 2~4 分钟 |
| 16GB 上会不会 OOM | `memory_optimization=True` 时大概率不 OOM（预估峰值 12~14GB）；`False` 时**必炸**（T5 10.8 + DiT 9.7 = 20.5GB）。真正的坑是 VAE decode |

---

## 1. 产物与目录

| 用途 | 路径 |
|---|---|
| 仓库 | `C:\Users\27236\Desktop\vla\third_party\YUME`（`git clone --depth 1`） |
| 权重 | `C:\Users\27236\Desktop\vla\models\Yume-5B-720P`（32.0GiB，apache-2.0） |
| 推理 venv | `C:\Users\27236\Desktop\vla\.venv_yume`（Python 3.10.11，**没动**主流程 `.venv`） |
| 下载 venv | `C:\Users\27236\Desktop\vla\.venv_hf`（huggingface_hub 2.4 + hf-xet 1.8） |
| **推理入口** | `third_party\YUME\yume_cli.py` |
| 自检 | `yume_cli.py --preflight` |
| VAE 探针 | `third_party\YUME\tools\vae_probe.py` |
| 下载工具 | `tools\download_weights.py`、`tools\download_loop.ps1` |

权重精确大小（HF API `tree/main?recursive=true`）：

```
diffusion_pytorch_model.safetensors   20,801,593,664 B = 19.373 GiB   (835 tensors, 全部 F32, 5.20B 参数)
models_t5_umt5-xxl-enc-bf16.pth       11,361,920,418 B = 10.582 GiB
Wan2.2_VAE.pth                         2,818,839,170 B =  2.625 GiB
google/umt5-xxl/tokenizer.json            16,837,417 B
```

---

## 2. 可复现安装命令（已实测成功）

```powershell
cd C:\Users\27236\Desktop\vla

# 仓库
git clone --depth 1 https://github.com/stdstu12/YUME.git third_party\YUME

# 推理环境（新建，不碰 .venv）
python -m venv .venv_yume
.\.venv_yume\Scripts\python.exe -m pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cu128     # Blackwell sm_120 必须 cu128，cu121 会 no kernel image
.\.venv_yume\Scripts\python.exe -m pip install `
    diffusers==0.32.0 transformers==4.46.1 tokenizers==0.20.1 safetensors==0.5.3 `
    huggingface-hub==0.26.1 sentencepiece==0.2.0 numpy==1.26.3 pillow==10.2.0 `
    easydict einops flask flask-cors ftfy regex imageio==2.36.0 imageio-ffmpeg==0.5.1 `
    av==13.1.0 tqdm psutil accelerate scipy protobuf pyyaml peft==0.13.2
```

**不需要** `pip install .`：仓库的 `run_oneclick_debug.bat` 本身也是走 `PYTHONPATH`（bat 第 126/160 行）。
**不需要** triton / flash-attn / apex / bitsandbytes：

- `wan23/modules/attention.py:4-14` 把 `flash_attn` / `flash_attn_interface` 包在 try/except 里，
  `attention()`（同文件 148-176 行）在没有 flash-attn 时回落到
  `torch.nn.functional.scaled_dot_product_attention`。实测 `FLASH_ATTN_2_AVAILABLE=False` 全链路正常。
- `requirements-extra.txt` 里的 `liger_kernel` 会硬依赖 `triton`（Windows 无 cp310 wheel），
  这是 `pip install -r` 失败的**唯一**原因（`ERROR: No matching distribution found for triton>=2.3.1 (from liger-kernel)`）。
- `peft==0.13.2` 是必须的：`wan23/textimage2video.py:34` → `fastvideo/utils/checkpoint.py:7` → `from peft import ...`。
- torch 用 2.11.0（pip 当前 cu128 最新）而不是仓库钉的 2.5.0+cu121：2.5.0 不支持 sm_120。
  实测 transformers 4.46.1 / diffusers 0.32.0 / numpy 1.26.3 与 torch 2.11.0 **兼容**；
  `torch.load` 对 T5/VAE 的 `.pth`（纯 state_dict）在 2.11 的 `weights_only` 默认值下也正常。

---

## 3. 模型怎么加载（以及为什么必须改）

`wan23/textimage2video.py: class Yume`：

```
T5EncoderModel  <- models_t5_umt5-xxl-enc-bf16.pth (10.58 GiB) + google/umt5-xxl/ (tokenizer)
Wan2_2_VAE      <- Wan2.2_VAE.pth (2.63 GiB), vae_stride=(4,16,16)   # 时间 4x / 空间 16x
WanModel        <- config.json + diffusion_pytorch_model.safetensors
```

架构（`wan23/configs/wan_ti2v_5B.py`）：`dim=3072, layers=30, heads=24, ffn_dim=14336, in/out_dim=48,
patch_size=(1,2,2)`；`WanModel.__init__` 在 `wan23/modules/model.py:486-492` 额外造
`patch_embedding_2x/4x/8x/16x/2x_f`（checkpoint 里确实有这些 key，共 835 个张量），
`mask_token`/`sideblock` 由 `webapp_single_gpu.load_wan()`（231-239 行）**随机初始化**，
但采样时 `flag` 默认为真，这两个分支不参与推理，所以随机初始化无害。

**上游 `Yume.__init__` 在 32GB 内存机器上会炸**（这是本机（31.6GB RAM）必须修的地方）：

1. `self.model = WanModel.from_config(config_wan)` → 默认 fp32 = **20.8GB**
2. 然后 `self.model = WanModel.from_pretrained(checkpoint_dir)` → **又一个 20.8GB**
3. 再 `load_file(...)` 读 19.8GB safetensors → 第三次

峰值 ~41GB。`yume_cli.py: patch_frugal_yume_init()` 替换成等价但省内存的版本：

```python
cfg = WanModel.load_config(checkpoint_dir)
torch.set_default_dtype(torch.bfloat16)          # 建模型就是 bf16 = 9.7GB
model = WanModel.from_config(cfg)
with safe_open(path, framework="pt", device="cpu") as f:   # 逐张量搬运
    for k in model.state_dict():
        model.state_dict()[k].copy_(f.get_tensor(k))
self.model = model.to("cuda", dtype=torch.bfloat16)         # 立刻释放 CPU 副本
...
self.text_encoder = T5EncoderModel(...)                     # 之后再加载 10.58GB 的 T5
```

实测效果：加载阶段**峰值内存 ~11GB**（而不是 41GB），且 `835` 个张量全部校验（缺失/多余都打印）。

---

## 4. 动作接口到底是什么（带仓库行号）

### 4.1 三套东西，全是文本

| 项 | 取值 | 仓库位置 |
|---|---|---|
| Keys（键盘/平移） | `W` `A` `S` `D`、`W+A` `W+D` `S+A` `S+D`、`None` | `webapp_single_gpu.py:648-658` |
| Mouse（鼠标/旋转） | `→` `←` `↑` `↓`、`↑→` `↑←` `↓→` `↓←`、`·` | `webapp_single_gpu.py:660-670` |
| 数值参数 | `Actual distance moved:<d> at 100 meters per second.`<br>`Angular change rate (turn speed):<a>.`<br>`View rotation speed:<v>.` | `caption_re.txt`、`sample_5b.py:807/858` |

**鼠标是「原地转」不是「平移」**——这是关键语义，原文：

```python
# webapp_single_gpu.py:660-670
vocab2 = {
    "→": "The camera pans to the right (→).",
    "←": "The camera pans to the left (←).",
    "↑": "The camera tilts up (↑).",
    "↓": "The camera tilts down (↓).",
    ...
    "·": "The rotation direction of the camera remains stationary (·)."
}
```

而平移是 keys 负责（648-658 行）：`"W": "The camera pushes forward (W)."`、
`"S": "The camera pulls back (S)."`、`"A": "The camera moves to the left (A)."`、
`"D": "The camera moves to the right (D)."`。
→ 所以 `turn_right` → mouse `→`，`move_forward` → key `W`，映射是对的。

拼装方式（673-683 行）：`final_prompt = "First-person perspective." + key句 + mouse句 + <用户 prompt>`。

完整训练格式（`caption_re.txt`，`sample_5b.sh` 就用它）：
```
First-person perspective. The camera pushes forward (W). The rotation direction of the camera
remains stationary (·). Actual distance moved:4 at 100 meters per second. Angular change rate
(turn speed):0. View rotation speed:0. <场景描述>
```

注意：`webapp_single_gpu.py` **没有**拼这三个数值参数（只拼两句 camera 描述）；
`README.md:72-78` 明确说这三个数会影响结果、建议 0.1~10。
`yume_cli.py` 默认按训练格式补上（4/0/0），`--omit_numeric_params` 可复现 webapp 的行为。

### 4.2 一个 chunk 只能一个离散键

模型**没有逐帧动作序列输入**。所以 `--actions "W:10,A:10"` 这种时间线必须压成「每 chunk 一个键」。
`yume_cli.py` 的规则（写死在 meta.json 里）：

1. 按帧展开时间线，缺的尾部用最后一个状态补齐（"hold last"）
2. 每个 chunk（默认 33 帧）内**多数票**决定 `(keys, mouse)`
3. 票数、`camera_sentence`、`prompt_passed_to_model`、`final_prompt` 全部写进 `meta.json`

例：`--frames 66 --actions "W:10,A:10"` → chunk0 = `A|none`（A 23 票 vs W 10 票），chunk1 同；
`--actions "W|right:33,S:33"` → chunk0 `W|→`，chunk1 `S|·`。

---

## 5. 一次调用能生成多少帧 / 必须 33 吗 / fps

### 5.1 帧数公式（代码推导，已核对）

- `webapp_single_gpu.py:585`：`latent_frame_zero = (frame_zero - 1) // 4 + 1`
- Wan2.2 VAE 时间压缩 4x 且首帧特殊：`decode(T 个 latent) = 4*(T-1)+1` 帧
- 所以 **`frame_zero` 必须是 4n+1**，否则 decode 出来的帧数对不上：
  - `frame_zero = 33` → `lfz = 9` → decode 出 `4*8+1 = 33` 帧 ✅
  - 仓库另一条路径 `sample_5b.py:875-876` 写的是 `frame_zero=32, latent_frame_zero=8`，
    实际只 decode 出 `4*7+1 = 29` 帧（代码用切片 `[-32:]` 兜住了，不报错但悄悄少 3 帧）。
    **所以不要用 32，用 33。**

### 5.2 边界

| | 值 | 依据 |
|---|---|---|
| 结构最小 | 5 帧（`lfz=2`） | `frame_zero=5` 时 head 16 帧→4 latent，`16+5`→6 latent，公式自洽；但严重 OOD，别用 |
| **有效/推荐** | **33 帧** | webapp 默认路径 + `frame_zero=33` 能整除；`README:55` 说「每行控制 2 秒视频」→ 33/16fps=2.06s |
| 单 chunk 上限 | 建议 ≤65 帧（`lfz=17`，21 latent，18,480 token） | token 数已接近作者验证过的「续帧 22 latent / 19,360 token」，再大就是未验证区 |
| 视频总长 | 无上限 | 靠 chunk 循环，每 loop 追加 33 帧 |

### 5.3 fps

`wan_ti2v_5B.py` 里 `sample_fps = 24`（Wan 默认），但 **Yume 的 33 帧 = 2 秒**，
所以播放帧率应为 **16fps**（`webapp_single_gpu.py:1423` 的 API 默认 `fps=16` 也印证）。
用 24fps 播 33 帧 = 1.375s，画面会偏快。

---

## 6. 闭环：上一段输出能不能当下一段输入

**官方支持，而且不是"最后一帧"，是"最后 49 帧的 latent"**。

- 首段：`create_video_from_image`（125-142 行）把输入图放到第 0 帧，复制 16 份做 head，
  再拼 33 帧 → `(C, 49, H, W)`；VAE 分两段编码 `[:16]`→4 latent、`[16:]`→9 latent，
  拼成 13 latent（628-635 行）。
- 采样只更新**尾部 9 个 latent**（784-817 行），头部 4 个 latent（=16 帧图）当条件；
  每步 `new_tail = tail + (σ_{i+1}-σ_i) * v_pred`（蒸馏模型 4~8 步就够）。
- 只 decode 尾部 9 latent → 33 帧新画面（831-832 行）。
- 续帧：`LAST` 里存 `last_model_input_latent` / `last_model_input_de`（872-876 行）；
  下次 `continue_from_last=True` 时（611-615 / 716-722 行）以它作上下文。
  更新方式（836-861 行）：
  ```python
  model_input_de = cat([model_input_de[:, :-frame_zero], video_tail_px[:, -frame_zero:]])
  ```
  即 **head16（还是最初那张图） + 上一段最后 33 帧**。
- 续帧时上下文会变成 **22 个 latent**（13 旧 + 9 新，770-777 行），token 数从 11,440 涨到 19,360，
  算力/显存都翻倍——这是 `--frames 96` 的成本所在。

`yume_cli.py` 在此之上加了跨进程闭环：`meta.json` 里 `closed_loop_state.npz` 存
`model_input_latent` / `model_input_de` / `frame_total`，下一次 `--resume_from <meta.json>` 直接续。

---

## 7. 实测数字

### 7.1 环境（实测）

| 项 | 值 |
|---|---|
| GPU | RTX 5060 Ti，capability **(12,0)** = sm_120，VRAM **15.93 GiB** |
| torch | 2.11.0+cu128（CUDA 12.8），`torch.cuda.is_bf16_supported()=True` |
| bf16 matmul | 正常（1024² 矩阵乘验证通过） |
| flash-attn | 未安装，`FLASH_ATTN_2_AVAILABLE=False`，SDPA 回退可用 |
| 全链路 import | `import wan23` / `import webapp_single_gpu` 均通过 |

### 7.2b T5 文本编码器（实测，`tools/t5_probe.py`，真实权重 10.58GB，纯 CPU）

| 阶段 | 输入 | 耗时 | 备注 |
|---|---|---|---|
| 加载 `models_t5_umt5-xxl-enc-bf16.pth` | torch.load + load_state_dict | **140.2 s** | 当时系统 RAM 只剩 1.2GB（被其他进程占满）→ 在 swap，**这是内存压力下的数字，正常应 30~60s** |
| encode（CPU，bf16） | 59 token 的完整 prompt | **9.5 s** | 输出 `(59, 4096)` bf16，数值正常 |

**结论**：
1. `torch.load` 在 torch 2.11 的 `weights_only` 默认值下**没问题**（`.pth` 是纯 state_dict）。
2. T5 在 CPU 上编码只要 9.5s → 如果内存吃紧，可以用 `t5_cpu=True` + 不打 `text_encoder.model.to("cuda")`
   （需要给 webapp 打 2 行补丁），换来：显卡峰值从 ~12.4GB 降到 ~11GB，**且 DiT 完全不用搬到 CPU**，
   RAM 峰值从 ~21GB 降到 ~10.6GB。代价是每 chunk 多 9.5s。
3. **RAM 是这台机器最紧的资源**：T5 加载时只看得到 1.2GB 可用，意味着真推理时如果同时开着浏览器/
   其他 agent，DiT(9.7GB) 在 CPU↔GPU 之间搬运会疯狂 swap。

### 7.2 VAE（实测，`tools/vae_probe.py`，真实权重 2.63GB）

| 阶段 | 输入 | 耗时 | 峰值显存 |
|---|---|---|---|
| 加载 | `Wan2.2_VAE.pth` | **2.3 s** | 2.63 GB |
| encode | 16 帧 704×1280（head） | **3.9 s** | **9.86 GB** |
| encode | 33 帧 704×1280（tail） | **9.5 s** | **10.03 GB** → latent `(48,9,44,80)` |
| decode | 9 latent → 33 帧 | **>300 s 未完成（被杀）** | — |

⚠️ **decode 是 16GB 卡上的真正地雷**：`vae.decode()` 一次解 33 帧 704×1280 时全分辨率特征图
是 `128ch × 704×1280 × 33 ≈ 15GB` 量级，Windows/WDDM 不会抛 OOM，而是悄悄溢出到共享显存 →
看起来「假死」（我的 5 分钟没跑完就是这么来的）。
破解：`vae_memory_optimization=True` → 走 `tiled_decode_overlap()`（`webapp_single_gpu.py:370-551`，
UI 传参在 829-830 行），分 5 块 + 32px 重叠解码。
**`yume_cli.py` 已把 `vae_memory_optimization` 默认设为 True。**

另外：VAE encode 就是那个 15.8GB 的元凶（16 帧 9.86GB / 33 帧 10.03GB，两次连续跑不 empty_cache 会叠到 15.8GB）。
所以 `memory_optimization=True` 在 `webapp_single_gpu.py:601-608` 先把 transformer 挪回 CPU 再 encode，
**这一步在 16GB 上是必需的**。

### 7.3 DiT / 单 chunk 墙钟（未实测，估算）

权重没下完，DiT 从未加载过，所以下表是**估算**，标注了依据：

| 阶段 | 估算 | 依据 |
|---|---|---|
| DiT 加载（bf16 需从 19.37GiB fp32 逐张量转换） | 40~90 s | 磁盘 20.8GB + PCIe 上传 9.7GB |
| chunk1 打分（13 latent = 11,440 token，6 步） | 15~40 s | 每步 ≈ 2·5.2e9·11440 ≈ 119 TFLOP，6 步 ≈ 714 TFLOP；5060 Ti bf16 实测有效 ~40-80 TFLOP/s |
| 续帧打分（22 latent = 19,360 token，6 步） | 25~70 s | token 数 ×1.69 |
| VAE encode（16+33 帧） | **13.4 s（实测 3.9+9.5）** | 实测 |
| VAE decode（tiled） | 10~40 s | 未实测 |
| **首 chunk 合计** | **30~60 s** | |
| **96 帧（3 chunk）合计** | **2~4 分钟** | |

### 7.4 16GB 显存预算（`--frames 96 --chunk_frames 33`）

| 阶段 | 显存 | 说明 |
|---|---|---|
| 加载后 | DiT 9.7GB + VAE 2.63GB | T5 在 CPU（10.58GB RAM） |
| VAE encode（transformer 已挪 CPU） | **10.03GB（实测）** | 独占，不与 DiT 叠加 |
| chunk1 打分（13 latent） | 9.7 + 1.5~2.5 = **11.2~12.2GB** | 估 |
| chunk2/3 打分（22 latent） | 9.7 + 2.5~4.0 = **12.2~13.7GB** | 估，token ×1.69 |
| VAE decode（**开 tiling**） | 9.7(已被挪走) → 2.63 + tiling 缓冲 ≈ 4~6GB | 必须开 |
| **预估峰值** | **12~14GB** | < 15.93GB，**大概率不 OOM** |

- 前提：`memory_optimization=True`（否则 T5 10.58 + DiT 9.7 = **20.5GB 必炸**）
- 前提：`vae_memory_optimization=True`（否则 decode 假死）
- 残余风险：Windows 的显存碎片/WDDM 共享内存会让「估算 13.7GB」变成「实际 15.xGB」，
  建议先跑 33 帧确认峰值，再上 96 帧。

### 7.5 内存（RAM）

| 阶段 | 峰值 | 说明 |
|---|---|---|
| frugal 加载 | ~11GB | DiT bf16 9.7GB 逐张量搬运后立刻上卡，再加载 T5 10.58GB |
| 生成时（memory_optimization） | **~21GB** | DiT 9.7GB 回 CPU + T5 10.58GB 常驻 CPU |
| 本机总量 | 31.6GB | 只够「刚好」，同时开浏览器/其他 agent 会 swap，swap 时 DiT 每次 CPU↔GPU 搬运会明显变慢 |

可选优化（没实现，因为要改仓库代码）：`t5_cpu=True` + 不打 `text_encoder.model.to("cuda")`，
RAM 峰值降到 10.58GB、显卡峰值降到 ~12.4GB，代价是 T5 在 CPU 上编码慢（估 20~60s/次）。

---

## 8. 踩坑清单（按发生顺序）

1. **cu121 装不了**：仓库钉 `torch==2.5.0+cu121`，但 cu121 没有 sm_120 kernel → 必须 cu128。
2. **flash-attn 编译失败不需要硬扛**：`attention.py` 有 SDPA 回退，直接不装。
3. **`pip install -r requirements.txt` 必失败**：`liger_kernel` 拉 `triton`（Windows 无 cp310 wheel）。
   按 `requirements-extra.txt` 那套（已经把 triton/flash-attn/bitsandbytes/gradio 注释掉）再手工去掉
   `liger_kernel` 即可；另外 `peft` 必须单独装。
4. **`pip install .` 不需要**：bat 本身就是 PYTHONPATH 方案，`pip install -e .` 失败也能跑。
5. **32GB 内存装不下上游加载流程**：见 §3，必须 frugal 加载。
6. **`webapp_single_gpu.py` 的 `frame_zero` API 默认值是 32**（1426 行），
   但 32 不是 4n+1，decode 只出 29 帧。**用 33**。
7. **HF 下载在本机基本不可用**：直连 1.8MB/s 单连接；8 并发 hf-mirror 5.6MB/s；32 并发直连 2MB/s；
   走本地 Clash（127.0.0.1:7897）单连接 1.33MB/s、8 并发反而 0.95MB/s（节点带宽封顶）；
   hf-xet 会突然「TLS EOF」然后 0 byte/s 卡死（`.incomplete` 文件时间戳停住）。
   最终 33GB 只下到 6.6GB。**这是本次唯一没跨过去的坎。**
   绕法（未验证完）：HF token（未登录配额低）、或在能跑满速的网络环境预下载后拷贝。
8. **`torch.cuda.amp.autocast` FutureWarning**：torch 2.11 弃用但可用，不影响。
9. **`expandable_segments not supported on this platform`**：Windows 不支持，忽略；
   不要指望它缓解碎片。
10. **PowerShell 的 `Get-Content -Raw | Set-Content -Encoding utf8` 会毁掉 UTF-8 中文/箭头字符**
    （按 GBK 解码再写 UTF-8 → 乱码）。`yume_cli.py` 里所有非 ASCII 已改成 `\u2192` 这类转义，
    文件现在 **纯 ASCII**，怎么倒腾都不会坏。

---

## 9. 权重到位后怎么跑（一条命令）

```powershell
cd C:\Users\27236\Desktop\vla\third_party\YUME
$env:PYTHONIOENCODING="utf-8"

# 0) 自检（不加载 DiT，几十秒）
..\..\.venv_yume\Scripts\python.exe yume_cli.py --preflight `
    --image ..\..\mj_test.png --prompt "A robot walking down a neon street" --outdir ..\..\outputs_yume

# 1) 最小推理：33 帧 / 6 步 / 704x1280
..\..\.venv_yume\Scripts\python.exe yume_cli.py `
    --image ..\..\mj_test.png --prompt "A robot walking down a neon street" `
    --actions "W|right:33" --frames 33 --steps 6 --seed 42 --outdir ..\..\outputs_yume

# 2) 96 帧（3 chunk，闭环续帧）
..\..\.venv_yume\Scripts\python.exe yume_cli.py `
    --image ..\..\mj_test.png --prompt "A robot walking down a neon street" `
    --actions "W|right:33,A:33,None|left:30" --frames 96 --steps 6 --seed 42 `
    --outdir ..\..\outputs_yume
```

输出：`yume_<ts>_s<seed>.mp4` + `yume_<ts>_s<seed>_frames/0000.png...` +
`yume_<ts>_s<seed>.meta.json` + `.latent.npz`（closed-loop 状态）。

`meta.json` 关键字段：`action_mapping`（每 chunk 的键/鼠标/票数/完整 prompt）、`generation`
（frames/fps/steps/chunk 数）、`timing`（model_load_s / chunk_times_s / total_wall_s）、
`memory`（peak_allocated_gb / peak_reserved_gb）、`closed_loop_state.npz`。

---

## 10. 待补（权重到位后 30 分钟内可完成）

1. 单 chunk（33 帧 / 6 步）**实测墙钟 + 峰值显存**（本文 §7.3/§7.4 是估算）
2. `--frames 96` 是否能真的跑完 + 峰值是否落在 12~14GB
3. T5 在 GPU 上 encode 一次的真实耗时（webapp 会临时把 T5 搬上卡）
4. 画面质量抽检：第 0 帧是否等于输入图（VAE round-trip 已经能确认编码/解码无损到肉眼）
5. 如果是「闭环世界模型」用途，验证 `--resume_from` 跨进程续帧的画面连续性
