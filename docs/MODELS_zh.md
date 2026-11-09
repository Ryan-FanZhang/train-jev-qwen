# 各尺寸的训练设置

所有命令都在 `nesi/` 目录里运行，用 `MODEL=` 选择尺寸。默认是 `2b`。

## 总览

| MODEL | 底座 | 起点 | 训练阶段 | 输出目录（`$PROJ_DIR/runs/`） |
|---|---|---|---|---|
| `0.8b` | Qwen/Qwen3.5-0.8B | 自己蒸馏 | distill → biz | `distill-0.8b/` → `my-biz-0.8b/` |
| `2b` | Qwen/Qwen3.5-2B | 自己蒸馏 | distill → biz | `distill-2b/` → `my-biz-2b/` |
| `4b` | Qwen/Qwen3.5-4B | shibing624/train-jev-4b-lora | biz | `my-biz/` |
| `27b` | Qwen/Qwen3.8-27B | shibing624/train-jev-27b-lora | biz | `my-biz-27b/` |

0.8B 和 2B 没有已发布的 train-jev 权重，所以先按 train-jev 的基础配方在 jev-distill 语料上蒸馏一个 epoch，
再做业务续训。四个底座都是 `Qwen3_5ForConditionalGeneration` 架构；0.8B 和 2B 的 `tokenizer.json` 完全一致。

## 两个阶段的参数（`nesi/03_train.sl`）

| 参数 | distill | biz |
|---|---|---|
| 数据 | jev-distill 训练集 | jev-distill + `--mix $BIZ_DATA/train.jsonl` |
| `LR` / `HEAD_LR` | 1e-4 / 2e-4 | 2e-5 / 4e-5 |
| `STEPS`（学习率曲线） | 0（一个 epoch，约 5,123） | 2048 |
| `MAX_STEPS`（停止） | 0 | 512 |
| `EVAL_EVERY` / `VAL_N` | 500 / 4000 | 256 / 2000 |
| `MIX_FRACTION` / `KIND_COUNTS` | - | 0.3 / `choice:40,noul:64,score:24` |
| 共同 | batch 128、LoRA r=16、`--rps_w 0.5`、`--perm 0.3`、`--placeholder_w 0.05`、`--max_len 2048`、seed 17 | 同左 |

distill 阶段结束后会自动在蒸馏语料的校准集上拟合 choice / score / noul 温度，并在 `test_set_30k`、`ood` 上记录通用能力
（`runs/distill-*/eval-distill/summary.json`）。

## 每个尺寸的资源设置（`nesi/config.sh`）

| MODEL | `MICRO_TOKENS` | gradient checkpointing | 训练时限 | 每卡申请内存 |
|---|---:|---|---|---:|
| `0.8b` | 16384 | 关 | 8 小时 | 48 GB |
| `2b` | 8192 | 关 | 8 小时 | 48 GB |
| `4b` | 8192 | 关 | 8 小时 | 48 GB |
| `27b` | 8192 | **开** | 16 小时 | 96 GB |

4B 在 `MICRO_TOKENS=8192`、不开 gradient checkpointing 时，单卡显存峰值是 88.6 GB，速度 3.45 秒/步。

## 常用覆盖（写在命令前）

| 变量 | 说明 |
|---|---|
| `GPU_SPEC=h100:2` | 单节点双卡（DDP）。续训时要保持同样的卡数 |
| `MICRO_TOKENS=4096` | 爆显存时减半 |
| `GC_ON=1` | 打开 gradient checkpointing，省显存、慢约 30% |
| `MEM_PER_GPU=160` | 任务因内存不足被杀时加大 |
| `TIME=04:00:00` | Slurm 时限 |
| `STEPS=2000 MAX_STEPS=2000` | distill 阶段先短跑 |
| `CKPT=<run>/step00256` | `eval` 指定评测的 checkpoint（默认最新的 `stepNNNNN`） |
| `BIZ_RUN_NAME=v2` | 换一个输出目录做对比实验 |

## 常见问题

| 现象 | 原因和处理 |
|---|---|
| `Requested mix fraction exceeds same-kind replacement capacity` | `128 × MIX_FRACTION` 超过了业务数据题型在 `KIND_COUNTS` 里的配额，把那个题型调大 |
| H100 上爆显存，日志里 `chunk_gated_delta_rule` 走参考实现 | flash-linear-attention 没加载上。重新 `pip install -r requirements-tested.txt`（`00_setup_env.sh` 已经这样做） |
| `KeyError: 'choice'`（评测示例） | 业务数据只有 noul，校准文件里没有 choice 温度。`04_eval.sl` 会把起点的 choice / score 温度合并进来 |
| `$'\r': command not found` | 脚本在 Windows 上被改成了 CRLF：`sed -i 's/\r$//' *.sh *.sl` |
| 任务超时 | 原样重新提交同一条命令，会从 `<run>/state` 接着训练 |

## 部署参考

| 尺寸 | bf16 权重 | 适合的硬件 |
|---|---:|---|
| 0.8B | 1.7 GB | 普通笔记本 CPU（8 GB 内存以上） |
| 2B | 4.5 GB | 8 GB 以上显卡，或 16 GB 内存的 CPU |
| 4B | 约 9 GB | 12 GB 以上显卡 |
| 27B | 约 55 GB | 80 GB 以上显卡 |

推理服务：`python -m trainjev.serve --base <底座目录> --adapter <run>/step00512`。服务没有鉴权，只在本机或私有网络里开放。
