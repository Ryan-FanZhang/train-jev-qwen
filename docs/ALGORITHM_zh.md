# 训练算法：一个训练步里发生了什么

下面按代码的实际执行顺序，以一条业务文本（这里用一条虚构的用户评价）为例，讲 train-jev 的一个训练步（step）。
代码在 [train-jev](https://github.com/shibing624/train-jev/tree/88bcc9d21eb4f3ec07da8183bb3f5898abe36bc5/trainjev)
的 `trainjev/` 目录下，每段都注明了出处。

## 总体思路

每条数据都被当作一道选择题：模型读完题目后，对每个选项各给出一个概率，训练目标是让这组概率尽量接近"标准答案的概率"。
业务续训就是把下面的循环重复 512 次（0.8B / 2B 的基础蒸馏约 5,123 次）：

```
抽 128 道题 → 拼成 prompt → 模型前向 → 决策头算概率 → 和目标比较算损失 → 反向传播 → 更新 LoRA 和决策头
```

## 第 1 步：抽一个 batch（`train.py` 的 `step_rows`）

每一步抽 **128 道题**，按题型固定配额抽取（`--kind_counts choice:40,noul:64,score:24`），保证每种题型都练到。

然后按 `--mix_fraction 0.3` 做替换：从 64 道 noul 题里随机挑 `round(128×0.3)=38` 道，换成业务数据。最终一个 batch 是：

| 来源 | 数量 | 作用 |
|---|---:|---|
| 业务数据 | 38 | 学新任务 |
| 蒸馏语料的 noul 题 | 26 | 保持是非题的通用能力 |
| 蒸馏语料的 choice 题 | 40 | 保持选择题的能力 |
| 蒸馏语料的 score 题 | 24 | 保持打分题的能力 |

这就是**回放（replay）**：每一步都混入旧任务，防止模型学了新的忘了旧的（灾难性遗忘）。
`128 × mix_fraction` 不能超过业务数据所属题型的配额，否则会报
`Requested mix fraction exceeds same-kind replacement capacity`。

另外还有两个细节：
- 选择题有 30% 的概率被打乱选项顺序，目标概率也跟着重排（`--perm 0.3`），让模型不依赖"答案常在 A"这类位置偏好。
- 没有判断信号的占位题，权重降到 0.05（`--placeholder_w`）。

## 第 2 步：拼成 prompt（`data.py`）

```
[kind] noul
[state] Ordered on Monday, promised delivery in two days, and it arrived nine days later. ...
[question] Is this customer review mainly about a delivery or availability problem (...)? ...
[options]
false
true
[decision]:
```

用 tokenizer 切成 token 序列。如果超过 2,048 个 token（`--max_len`），会**从正文中间截掉**一段，问题和选项始终保留完整。

## 第 3 步：模型前向（`modeling.py` 的 `last_hidden`）

```
token 序列 → Qwen 底座（权重冻结）+ LoRA → 每个位置一个隐藏向量 h
```

- **底座冻结**：Qwen 原有的参数完全不更新。
- **LoRA**：在注意力层和 MLP 的投影矩阵（`q_proj`、`v_proj`、`out_proj` 等）旁边各加一对秩为 16 的小矩阵。
  训练只更新这些小矩阵，4B 上约 3,050 万参数，27B 上约 1.09 亿参数。
- **只取最后一个 token 的向量**：最后一个 token 在 `[decision]:` 末尾，此时模型已经读完整道题，它的隐藏向量 h 就是对整道题的理解。

128 道题会按长度分成若干 micro-batch（每块最多 `--micro_tokens` 个 token）分别前向，梯度累加。`micro_tokens` 太大会爆显存。

## 第 4 步：决策头算概率（`modeling.py` 的 `DecisionHead`）

决策头是一个 **24 行的线性层**，每一行对应一个"答案 token"：

| 槽位 | token | 题型 |
|---|---|---|
| 0–1 | `false` `true` | noul（是非题） |
| 2–7 | `0`–`5` | score（打分题） |
| 8–23 | `A`–`P` | choice（选择题） |

`logit = W · h + b`，只取当前题型的槽位做 softmax：

```
[W_false·h + b_false,  W_true·h + b_true]  →  softmax  →  [P(false), P(true)]
```

W 的 24 行不是随机初始化，而是复制底座 `lm_head` 里这些 token 对应的行。所以训练一开始，
决策头就等于"底座预测下一个词是 true 还是 false"，相当于从模型已有的语言能力出发。

## 第 5 步：算损失（`losses.py` 的 `kl_rps_batch`）

主损失是 KL 散度，衡量模型的概率 q 和目标概率 t 差多远：

```
KL(t ‖ q) = Σ  t_i · (log t_i − log q_i)
```

例：老师给了 high 置信度，目标 t = [0.03, 0.97]。
- 模型输出 q = [0.04, 0.96] → KL ≈ 0.0004，损失很小；
- 模型输出 q = [0.70, 0.30] → KL ≈ 1.03，损失很大。

用软标签而不是 0/1：low 置信度的样本目标是 0.65，模型会学到"边界样本不要太确定"，输出的概率也就更可信（ECE 低）。

打分题额外加 **0.5 × RPS**（`--rps_w 0.5`），比较累积概率：真实答案 4 分时，猜 3 分比猜 0 分罚得轻。

每道题的损失乘上它的 `weight`，对整个 batch 取平均。

## 第 6 步：反向传播与更新（`train.py` 主循环）

```
loss.backward()               → 只算 LoRA 和决策头的梯度
all_reduce_grads()            → 多卡时把各卡梯度取平均
clip_grad_norm_(..., 1.0)     → 梯度总范数超过 1 就按比例缩小
opt.step()                    → AdamW（betas 0.9 / 0.98，weight decay 0）
```

| 阶段 | LoRA 学习率 | 决策头学习率 | 学习率曲线长度 | 实际步数 |
|---|---:|---:|---:|---:|
| 基础蒸馏（0.8B / 2B） | 1e-4 | 2e-4 | 一个 epoch | 一个 epoch |
| 业务续训（所有尺寸） | 2e-5 | 4e-5 | 2,048 | 512 |

学习率调度：前 3% 的步数线性预热，然后余弦下降到设定值的 0.02 倍。业务续训的曲线按 2,048 步设计，
在第 512 步停止，所以学习率只下降了一小段。

## 第 7 步：验证与保存

- 每 `--eval_every` 步，在蒸馏语料的固定验证子集上算 KL，并保存 `stepNNNNN/`；
- 验证 KL 最低的那次另存为 `best/`。注意 `best/` 是按蒸馏语料选的，业务效果以 `04_eval` 在业务测试集上的结果为准；
- 每 25 步保存一次完整训练状态（参数、优化器、随机数状态），超时后重新提交就能接着跑。

## 第 8 步：训练之后（`evaluate.py`）

**温度校准**：在 calibration 集上找一个系数 T，让 `softmax(logits / T)` 的 KL 最小。只调一个数，不会过拟合；
它只改变概率的"刻度"，不改变哪个选项最大。

**评测**：在 test 集上算 top1、AUROC、KL、ECE。

## 一句话总结

> 用一段固定格式的文字把题目交给 Qwen，取它"读完题"那一刻的内部表示，再用一个从"true/false"词向量初始化的小线性层读出概率；
> 训练时只调 LoRA 小矩阵和这个线性层，让概率在 KL 意义上贴近老师给的软标签，同时混入旧题防止遗忘。
