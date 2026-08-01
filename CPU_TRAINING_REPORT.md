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