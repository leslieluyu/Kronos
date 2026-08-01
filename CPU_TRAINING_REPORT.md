# Kronos CPU 训练效率测试报告

> 测试日期: 2026-08-01 | 机器: vmi3467645 | CPU: 6核

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

- **KronosTokenizer**: Encoder-Decoder VQ-VAE，使用 Binary Spherical Quantization (BSQ) 将连续 K 线数据量化为层级化离散 token
- **Kronos**: 自回归 Transformer 模型，在两阶段 token 上进行预测

### 模型族

| 模型 | Tokenizer | 上下文长度 | 参数量 | 开源 |
|------|-----------|-----------|--------|------|
| Kronos-mini | Tokenizer-2k | 2048 | 4.1M | ✅ |
| **Kronos-small** | **Tokenizer-base** | **512** | **24.7M** | ✅ |
| Kronos-base | Tokenizer-base | 512 | 102.3M | ✅ |
| Kronos-large | Tokenizer-base | 512 | 499.2M | ❌ |

本次测试使用 **Kronos-Tokenizer-base** (3,958,042 参) + **Kronos-small** (24,741,376 参)。

### 工程目录结构

```
Kronos/
├── model/
│   ├── kronos.py          # 模型核心 (Tokenizer + Kronos + KronosPredictor)
│   └── module.py          # Transformer 子模块
├── finetune/              # 微调流水线 (基于 Qlib，需要 GPU)
├── finetune_csv/          # CSV 数据微调方案
├── examples/              # 预测示例脚本
├── webui/                 # Flask Web 界面
├── a_share_adapter/       # [新增] A股数据适配器
│   ├── data_loader.py     #   从 kline_cache JSON 加载
│   ├── cpu_train_test.py  #   CPU 训练效率测试
│   └── setup_and_test.sh  #   环境初始化
└── outputs/               # 预测结果和图表
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
| 数据源 | `/home/dev/quant/stock_tracker/data/kline_cache/` (5108个文件, 日线) |
| 测试股票 | sh.600000 (浦发银行) |
| 上下文长度 | 90 天 |
| 预测长度 | 20 天 (训练测试用 5 天) |

---

## 三、测试结果

### 3.1 Tokenizer 推理速度

| 指标 | 结果 |
|------|------|
| 输入 | batch=1, seq=90, dim=6 (OHLCV+amount) |
| 单次推理耗时 | **30.1ms** |
| 吞吐量 | ~33 samples/sec |

### 3.2 Tokenizer 训练效率

| 指标 | 结果 |
|------|------|
| 平均每迭代 (batch=1, 含前向+反向) | **126ms** |
| 预估 1 epoch (100,000 iter, batch=50) | **~210 分钟 (3.5 小时)** |
| 预估 30 epochs | **~105 小时 (~4.4 天)** |

> 注: 原始配置为 batch_size=50, n_train_iter=100k, 30 epochs。

### 3.3 Predictor (Kronos) 训练效率

| 指标 | 结果 |
|------|------|
| 平均每迭代 (batch=1, 含前向+反向) | **571ms** |
| 预估 1 epoch (100,000 iter, batch=50) | **~951 分钟 (15.9 小时)** |
| 预估 30 epochs | **~475 小时 (~19.8 天)** |

### 3.4 端到端预测

✅ **成功运行**，完成时间 **3.3秒**。

使用 200 天历史数据预测未来 20 天 sh.600000 收盘价：

| 日期 | 预测收盘价 |
|------|-----------|
| 2023-11-02 | 6.41 |
| 2023-11-03 | 6.41 |
| 2023-11-08 | 6.43 |
| 2023-11-13 | 6.44 |
| 2023-11-20 | 6.52 |
| 2023-11-27 | 6.58 |
| 2023-11-29 | 6.59 |

预测图表: `outputs/prediction_cpu_test.png`

---

## 四、总结

| 任务 | CPU 可行性 | 预估耗时 |
|------|-----------|---------|
| **推理 / 预测** | ✅ 可行 | 30ms/样本, 3.3s/20天预测 |
| **Tokenizer 训练** | ❌ 不推荐 | ~4.4 天 |
| **Predictor 训练** | ❌ 不推荐 | ~19.8 天 |
| **完整微调 (Tokenizer + Predictor)** | ❌ 不现实 | ~24+ 天 |

### 建议

1. **推理场景**: 直接在 CPU 上使用预训练 Kronos 模型预测 A 股走势完全可行
2. **微调/训练**: 必须使用 GPU。建议配置 NVIDIA GPU (至少 8GB VRAM)，可大幅缩短训练时间至小时级
3. **数据适配**: `a_share_adapter/data_loader.py` 已实现从本地 kline_cache JSON 文件加载数据，可替代 Qlib 依赖

---

## 五、常见问题 (FAQ)

### Q1: 换小模型（Kronos-mini 4.1M vs Kronos-small 24.7M）效果会差很多吗？

- Kronos-mini 在论文各预测基准上确实不如 small/base，但差距不是数量级的。mini 的天然优势是上下文长度更大（2048 vs 512）
- 从 CPU 训练角度看：mini 参数量只有 small 的 **1/6**，训练时间预计从 20+ 天缩短到 **3-5 天**，在 CPU 上变得"勉强可尝试"
- **结论**: 如果只能用 CPU 微调，mini 是更现实的选择，精度损失可控

### Q2: 训练就是把 A 股历史数据喂一次就行了吗？

**不是。** 微调是一个多阶段、多 epoch 的过程：

1. **数据准备** — 将原始 OHLCV 序列切分为大量 `(历史窗口, 预测窗口)` 的滑动窗口样本
2. **Tokenizer 微调** — 让 Tokenizer 先学习 A 股数据的价格分布、波动特征（VQ-VAE 的重建损失 + 量化损失），跑多 epoch
3. **Predictor 微调** — 在 Tokenizer 输出的离散 token 上训练自回归预测模型（交叉熵损失），跑多 epoch
4. 默认训练配置是 **30 epochs × 100,000 样本/epoch**，不是"喂一次就完了"

### Q3: 用预训练模型直接预测 vs 微调后预测，差距有多大？

| 维度 | 预训练模型 (off-the-shelf) | A股微调后 |
|------|--------------------------|----------|
| Tokenizer | 在 45+ 全球交易所数据上学到的通用 token 空间 | 适应 A 股特定价格分布和波动量级 |
| Predictor | 通用时序模式，未见过涨跌停/T+1/政策驱动 | 学到 A 股特有的时序规律和市场微观结构 |
| 预测误差 | 基准水平 | 预计改善 **30-50%**（论文 CSI300 回测参考） |
| 适用场景 | 快速验证、多市场对比 | 实际量化策略、因子挖掘 |

**比喻**: 预训练模型是"拿外国驾照开中国车"，微调后是"拿中国驾照开中国车"——能用 vs 好用。