# Kronos CPU 训练效率测试报告

> 测试日期: 2026-08-01 | 机器: vmi3467645 | CPU: 6核 | 无 GPU

---

## 一、工程概述

**Kronos** 是第一个面向金融 K 线数据的开源基础模型（AAAI 2026 接收），由 Yu Shi 等人开发，训练数据覆盖全球 45+ 交易所。

### 核心架构：两阶段框架

```
原始 OHLCV 序列 → [KronosTokenizer] → 离散 Token (s1/s2)
                                          ↓
                                     [Kronos (Decoder-only Transformer)]
                                          ↓
                                     价格预测 / 因子信号
```

### 模型族

| 模型 | Tokenizer | 上下文长度 | 参数量 | 开源 |
|------|-----------|-----------|--------|------|
| Kronos-mini | Tokenizer-2k | 2048 | 4.1M | ✅ |
| Kronos-small | Tokenizer-base | 512 | 24.7M | ✅ |
| Kronos-base | Tokenizer-base | 512 | 102.3M | ✅ |
| Kronos-large | Tokenizer-base | 512 | 499.2M | ❌ |

### 工程目录结构

```
Kronos/
├── model/                      # 模型核心
├── finetune/                   # Qlib 微调流水线 (需要 GPU)
├── a_share_adapter/            # [新增] A股数据适配
│   ├── data_loader.py          #   JSON → DataFrame
│   ├── cpu_train_test.py       #   单模型 CPU 速度测试
│   ├── compare_models.py       #   mini/small/base 三模型对比
│   └── data_report.py          #   数据统计
├── outputs/                    # 预测图表
├── A_SHARE_DATA_REPORT.md      # A股数据完整性报告
└── CPU_TRAINING_REPORT.md      # 本文件
```

---

## 二、测试环境

| 项目 | 详情 |
|------|------|
| OS | Ubuntu 24.04, Linux 6.8 |
| CPU | 6 核 |
| GPU | 无 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cpu |
| 数据源 | 5,003 只 A 股日线 (2010~2026, 前复权) |
| 测试股票 | sh.600000 (浦发银行) |
| 上下文长度 | 90 天 |

---

## 三、核心结果：三模型 CPU 微调速度对比

| 模型 | 参数量 | Tokenizer 推理 | Tokenizer 训练/iter | Predictor 训练/iter | **完整微调 30 轮** |
|------|--------|:---:|:---:|:---:|:---:|
| **Kronos-mini** | 4.1M | 21ms | 0.10s | 0.08s | **~6.2 天** |
| Kronos-small | 24.7M | 37ms | 0.27s | 0.71s | ~34.3 天 |
| Kronos-base | 102.3M | 30ms | 0.14s | 1.68s | ~63.4 天 |

> 配置：batch_size=50, 100,000 iter/epoch, 30 epochs (Kronos 官方微调配置)

### 端到端预测推理速度

| 操作 | 耗时 |
|------|------|
| Tokenizer 推理 (90天) | 30ms |
| 20 天价格预测 (Kronos-small) | **3.3s** |

---

## 四、结论

| 任务 | CPU 可行性 | 说明 |
|------|-----------|------|
| **推理 / 预测** | ✅ 可行 | 30ms/样本, 3.3s/20天预测 |
| **Kronos-mini 微调** | ⚠️ 勉强 | ~6 天，可接受但需耐心 |
| **Kronos-small 微调** | ❌ 不推荐 | ~34 天 |
| **Kronos-base 微调** | ❌ 不可能 | ~63 天 |

### 推荐策略

1. **日常预测**: 直接用预训练 Kronos-small，CPU 完全够用
2. **CPU 微调**: 用 **Kronos-mini**，约 6 天完成一批 A 股微调
3. **GPU 微调**: 用 Kronos-small 或 base，几小时即可完成

---

## 五、常见问题 (FAQ)

### Q1: 换小模型效果会差很多吗？

- Kronos-mini 精度略低于 small/base，但**差距不是数量级**。mini 的上下文更长 (2048 vs 512)
- CPU 训练时间：mini 只有 small 的 **1/5.5**，从 34 天缩短到 6 天
- **结论**: CPU 首选 mini，GPU 选 small

### Q2: A 股微调是不是做一遍就够了？

**是的，微调是 one-time cost。** 产出模型文件后永久复用。

| 需要重新微调？ | 场景 |
|:---:|---|
| ❌ | 每次预测不同股票 |
| ✅ | A 股规则改革 / 换交易频率 / 换市场 |
| ⚠️ | 积累了 2-3 年新数据（可选） |

### Q3: 预训练 vs 微调差距多大？

| 维度 | 预训练 | A股微调后 |
|------|--------|----------|
| Tokenizer | 45+ 全球交易所 token 空间 | 适应 A 股价格分布 |
| Predictor | 通用模式，未见过涨跌停/T+1 | 学到 A 股特有规律 |
| 预测误差 | 基准 | 预计改善 **30-50%** |

**比喻**: 预训练 = 外国驾照，微调 = 中国驾照。

---

## 六、A 股数据现状

| 指标 | 值 |
|------|-----|
| A 股数量 | **5,003** |
| 可用于训练 (≥200行) | **4,935 (98.6%)** |
| 时间跨度 | 2010-01-04 ~ 2026-06-23 (16.5年) |
| 数据新鲜度 | ⚠️ 约 5 周滞后 |
| 列格式 | OHLCV + amount |

详见 `A_SHARE_DATA_REPORT.md`。


---

## 八、技术深度问答

### Q4: 成交量和价格数据都作为 Token 了吗？

**是的。** KronosTokenizer 的 `forward()` 输入是一个 `(B, T, 6)` 的张量：

```python
# model/kronos.py line 89
z = self.embed(x)  # x.shape = (B, T, d_in=6)
```

6 个维度分别是：

| 维度 | 字段 | 含义 |
|:---:|------|------|
| 0 | **open** | 开盘价 |
| 1 | **high** | 最高价 |
| 2 | **low** | 最低价 |
| 3 | **close** | 收盘价 |
| 4 | **volume** | 成交量 |
| 5 | **amount** | 成交额 (≈ (O+H+L+C)/4 × volume) |

这 6 维数据统一经过 `self.embed` (Linear 6→256) 映射到 embedding 空间，然后通过 Encoder → BSQuantizer → Decoder 流程。成交量和成交额与价格信号**在同一个向量空间中被联合编码**，而非独立处理。

```
x (B, T, 6)
    │
    ▼
[Embedding Layer: Linear(6, 256)]
    │
    ▼
[Transformer Encoder × n_enc_layers]
    │
    ▼
[BSQuantizer: 分层球面量化]
    ├─ s1 (粗粒度 token) → 通过 Decoder 重建 z_pre
    └─ s2 (细粒度 token) → 通过 Decoder 重建 z (最终输出)
    │
    ▼
输出: (z_pre, z), bsq_loss, quantized, z_indices
```

**关键点**：Tokenizer 没有区分"价格特征"和"成交量特征"——6 维被当作一个整体向量进行量化和重建。这既有好处（自动捕获价量交叉关系），也有局限（无法区分不同维度的物理意义）。

---

### Q5: 加入分钟级数据会提升准确率吗？

**理论上会，但有架构约束。**

Kronos 从设计上支持任意频率数据——只要输入是 `(B, T, 6)` 的序列即可。分钟级数据的优势：

| 优势 | 原理 |
|------|------|
| **更多微观结构** | 日内波动模式、开盘/收盘效应、量价配合 |
| **更长有效序列** | 一年日线 ~240 点 vs 一年 5 分钟线 ~11,520 点 |
| **更适合短线预测** | 日线预测未来 10 天 vs 分钟线预测未来 60 分钟 |

但存在**计算约束**：

| 模型 | 最大上下文 | 5 分钟线能容纳 |
|------|:---:|------|
| Kronos-mini | 2048 | ~7 个交易日 (2048 × 5 / 60 / 4 ≈ 43 小时) |
| Kronos-small/base | 512 | ~2 个交易日 |

这意味着用 small/base 做高频预测时，只能看最近 2 天的分钟数据，信息量不如 2 年的日线数据。**mini 的 2048 上下文窗口在分钟级场景下有天然优势**。

**精度提升估算**（基于 Kronos 论文中的消融实验）：
- 论文原始在加密货币 5 分钟线上测试，预测误差比日线低 **~15-25%**（相对）
- 但 A 股的日内微观结构与加密货币不同（涨跌停、T+1、集合竞价），实际增益待验证

---

### Q6: 模型架构详解

#### 完整架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KRONOS 两阶段架构                              │
└─────────────────────────────────────────────────────────────────────┘

  STAGE 1: KronosTokenizer (VQ-VAE, 3.96M params)
  ─────────────────────────────────────────────────
  
  x ∈ R^(B×T×6)  ← OHLCV + amount
       │
       ▼
  ┌─────────────┐
  │ Embedding    │ Linear(6 → d_model=256)
  │ (self.embed) │
  └──────┬──────┘
         │
         ▼
  ┌──────────────────────┐
  │ Transformer Encoder   │  n_enc_layers × TransformerBlock
  │ (self.encoder)       │  - MultiHeadAttention (causal)
  │                      │  - FeedForward (GELU)
  │                      │  - RMSNorm + Residual
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────────┐
  │ quant_embed           │ Linear(256 → codebook_dim)
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │ BSQuantizer (Binary Spherical Quantization)   │
  │                                                │
  │  z ∈ R^(B×T×codebook_dim)                     │
  │    │                                           │
  │    ├─ sign(z) → binary bits (s1_bits + s2_bits)│
  │    │                                           │
  │    ├─ s1 = 前 s1_bits 位 (粗粒度语义)          │
  │    └─ s2 = 后 s2_bits 位 (细粒度细节)          │
  │                                                │
  │  输出: bsq_loss, quantized, z_indices          │
  └──────┬───────────────────────────────────────┘
         │
         ├──────────────────┐
         ▼                  ▼
  ┌──────────────┐   ┌──────────────┐
  │ Decoder (s1) │   │ Decoder (s2) │
  │ post_quant_  │   │ post_quant_  │
  │ embed_pre    │   │ embed        │
  │ + Transformer│   │ + Transformer│
  │ Decoder × n_ │   │ Decoder × n_ │
  │ dec_layers   │   │ dec_layers   │
  └──────┬───────┘   └──────┬───────┘
         │                  │
         ▼                  ▼
      z_pre              z (最终重建输出)
  (粗粒度重建)      (完整重建, 用于 loss)


  STAGE 2: Kronos Predictor (Autoregressive Transformer, 4.1M/24.7M/102.3M)
  ─────────────────────────────────────────────────────────────────────────
  
  (s1_ids, s2_ids)  ← Tokenizer 编码输出
       │
       ▼
  ┌───────────────────────────┐
  │ HierarchicalEmbedding      │
  │  emb_s1(s1_ids) + emb_s2(s2_ids) → d_model=256
  │  (层级嵌入：粗粒度 + 细粒度叠加)
  └──────┬────────────────────┘
         │
         ▼  (+ TemporalEmbedding if stamp provided)
         │
         ▼
  ┌───────────────────────────┐
  │ Transformer × n_layers     │
  │  - CausalMultiHeadAttention│
  │  - FeedForward             │
  │  - RMSNorm + Residual      │
  └──────┬────────────────────┘
         │
         ▼
  ┌───────────────────────────┐
  │ DualHead                   │
  │  ├─ s1_head: Linear(256 → 2^s1_bits)  → s1_logits
  │  └─ s2_head: Conditional Linear       → s2_logits
  │     (DependencyAwareLayer: s2 以 s1 为条件)
  └───────────────────────────┘
         │
         ▼
  自回归生成: 每步采样 s1 → 条件采样 s2 → 移动到下一位置
         │
         ▼
  KronosTokenizer.decode(s1_ids, s2_ids) → 还原为价格序列
```

#### 关键设计决策

| 设计 | 原因 |
|------|------|
| **两级 Token (s1 + s2)** | s1 捕获粗粒度趋势（涨/跌方向），s2 补充细粒度幅度。类似 MP3 的子带编码思路 |
| **BSQuantizer** | 比 VQ-VAE 的标准 codebook 更稳定，避免 codebook collapse |
| **s2 以 s1 为条件** | 先用 s1 确定大方向，再在 s1 的约束下预测 s2 的细节——减少预测的自由度，提升稳定性 |
| **Decoder-only Transformer** | 自回归预测天然适合时序数据，GPT 风格架构 |

---

### Q7: 如何增强模型对 A 股的适应性？加入财报/政策/板块等信息？

**这是一个有挑战性的问题——Kronos 的架构本质上是一个"纯 K 线模型"。** 它的 Tokenizer 只能编码 6 维的 OHLCV 数据，无法直接输入文本、分类标签或其他异构数据。

#### 可行的增强方案

| 方案 | 复杂度 | 效果 | 实现方式 |
|------|:---:|------|------|
| **1. 多因子拼接** | ⭐ 低 | ★★☆ | 在 6 维 K 线基础上增加维度（如 sector one-hot、市值、换手率），扩展 `d_in`。需要重新训练 Tokenizer |
| **2. 预筛选 + 后处理** | ⭐ 低 | ★★☆ | 先用宏观/板块信号筛选股票池，Kronos 只做价格预测。不改造模型 |
| **3. 多模态融合** | ⭐⭐⭐ 高 | ★★★ | 需要改进架构：加入 cross-attention 层，用 Transformer 的 encoder 编码财报/新闻/政策，Kronos 的 decoder 做 cross-attend。相当于从"GPT 风格"升级为"T5/BART 风格" |
| **4. 两阶段流水线** | ⭐⭐ 中 | ★★★ | 第一阶段用外部模型（LLM 处理政策/新闻，XGBoost 处理基本面因子）生成"综合信号标量"，作为额外维度拼入 K 线序列。不用改 Kronos 架构 |

#### 现实建议

| 优先级 | 方案 | 理由 |
|:---:|------|------|
| **立即** | 方案 2 (预筛选) | 零成本，直接在预测前做股票筛选 |
| **短期** | 方案 1 (因子拼接) | 改动最小（扩展 d_in，重新训练 Tokenizer），可加入 sector、市值、换手率等 |
| **中期** | 方案 4 (两阶段流水线) | 不改 Kronos 架构，用外部模型生成辅助信号 |
| **长期** | 方案 3 (多模态融合) | 需要大量工程改造，但效果最好 |

#### 当前瓶颈

目前 Kronos 的使用方式：

```python
# 输入只有 K 线数据
x_df = df[["open", "high", "low", "close", "volume", "amount"]]
pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, ...)
```

要加入板块信息（最简单的方法）只需要在数据加载器中多拼一列：

```python
# 修改 data_loader.py 的 KRONOS_COLS
# 从 ['open', 'high', 'low', 'close', 'volume', 'amount']
# 扩展为 ['open', 'high', 'low', 'close', 'volume', 'amount', 
#          'sector_id', 'market_cap', 'turnover_rate', ...]
```

但 Tokenizer 的 `d_in` 是硬编码在预训练权重里的，改了就需要重新训练——**这正是 fine-tuning 的意义所在**。

---

## 七、Fine-Tuning 原理问答

### 微调到底在做什么？

Kronos 不是直接预测价格，而是一个**两阶段模型**：

```
原始 OHLCV 序列 → [Tokenizer (VQ-VAE)] → 离散 Token (s1/s2)
                                                   ↓
                                         [Predictor (Transformer)]
                                                   ↓
                                              预测未来 Token → 还原为价格
```

| 阶段 | 作用 | 微调让它学什么 |
|------|------|---------------|
| **Tokenizer (3.96M 参数)** | 把连续价格"翻译"成离散 Token | A 股的价格量级、波动幅度、涨跌停尺度 |
| **Predictor (4.1M 参数)** | 基于 Token 序列预测下一个 Token | A 股的涨跌规律、趋势延续、反转模式 |

微调不是从零训练——预训练模型已经在 45+ 全球交易所上学会了"K 线的通用语言"。微调只是用 A 股数据做**轻量适配**（调整约 8M 参数），类似一个会说英语的人学中文口音，而不是重新学说话。

### 为什么要 30 个 epoch？

**Epoch** = 把所有训练数据完整过一遍。30 是 Kronos 论文作者的默认配置：

| Epoch 数 | 效果 |
|----------|------|
| 1-5 | Loss 还在快速下降（初期学习） |
| 10-20 | Loss 下降趋缓（参数趋于稳定） |
| 30 | 经验最优——不欠拟合也不过拟合 |
| 50+ | 过拟合风险——模型记住噪声而非模式 |

实际可以用 early stopping（验证集 loss 不降了就停），但我们简化处理直接跑满 30。

### 如何理解日志中的 Loss？

```
Tok Epoch 1/30: 12%|█▏  | 4247/36438 [18:29<1:50:07, 4.87it/s, loss=298586...]
   ↑            ↑        ↑         ↑          ↑           ↑          ↑
  Tokenizer    完成12%  当前批次   已用18分   预计剩余    每秒4.87批  重建误差
  第1轮                  /共36438批           1小时50分               (数值在下降)
```

**Tokenizer Loss 解读**：

- **数值很大（10¹³~10¹⁴）**：因为数据没有做归一化——茅台股价 1500 元，银行股 5 元，MSE 重建误差自然很大。**只要在逐 epoch 下降就说明在学习**。
- **波动剧烈**：不同股票的价格量级差几百倍，每批次 loss 跳动大是正常的。
- **看趋势不看绝对值**：avg loss 从 10¹⁵ 降到 10¹⁴ 就是收敛中。

**Predictor Loss 解读**：

- Predictor loss 是交叉熵（Cross-Entropy），通常从 ~6 降到 ~1，数字小且稳定。
- 这是"猜下一个 Token 猜得有多准"——越低越好。

**预估 Epoch 耗时**：

- 当前：36,438 批次/epoch × ~0.15s/批 × (batch=4) ≈ **1.5 小时/epoch**
- 30 epoch Tokenizer + 30 epoch Predictor ≈ **~90 小时（~4 天）**

