# Session Summary — Kronos A-Share Fine-Tuning

> Date: 2026-08-01 ~ 08-02 | Machine: vmi3467645 (6-core CPU, no GPU)

---

## 一、任务目标

在 CPU 机器上运行 Kronos-Tokenizer-base 微调，使用 A 股历史 K 线数据（来自 `/home/dev/quant/stock_tracker/data/kline_cache/`），评估 CPU 微调可行性并落地全量训练。

---

## 二、完成的工作

### 2.1 环境搭建

- 创建 Python 3.12 venv (`Kronos/.venv/`)
- 安装 CPU PyTorch 2.13.0 + pandas、numpy、einops、huggingface_hub、matplotlib、tqdm、safetensors、baostock
- 从 HuggingFace Hub 下载 Kronos-Tokenizer-base + Kronos-small + Kronos-mini + Kronos-Tokenizer-2k

### 2.2 A 股数据适配

新创建 `a_share_adapter/` 目录：

| 文件 | 作用 |
|------|------|
| `data_loader.py` | 从 kline_cache JSON 文件加载 A 股数据，输出 Kronos 兼容的 DataFrame |
| `cpu_train_test.py` | 单模型 CPU 速度基准测试（Tokenizer 推理、训练迭代、Predictor 训练、端到端预测） |
| `compare_models.py` | mini / small / base 三模型 CPU 训练速度对比 |
| `data_report.py` | A 股数据完整性扫描脚本 |
| `train_mini.py` | 完整的两阶段微调流水线（Tokenizer + Predictor），支持 CLI 参数 |
| `setup_and_test.sh` | 一键环境初始化脚本 |

### 2.3 数据全景扫描

- 总 JSON 文件：5,108
- 有效 A 股：**5,003** (上证 2,230 + 深证 2,773)
- 时间跨度：2010-01-04 ~ 2026-06-23 (16.5 年)
- 可用于训练 (≥200行)：**4,935 只 (98.6%)**
- 列格式：OHLCV + amount (7~8 列，前复权)
- 数据新鲜度：约 5 周滞后（截至 2026-06-23）

详见 `A_SHARE_DATA_REPORT.md`。

### 2.4 CPU 微调速度基准

| 模型 | 参数量 | Tokenizer 训练/iter | Predictor 训练/iter | **完整微调 30 epochs** |
|------|--------|:---:|:---:|:---:|
| **Kronos-mini** | 4.1M | 0.10s | 0.08s | **~6.2 天** |
| Kronos-small | 24.7M | 0.27s | 0.71s | ~34.3 天 |
| Kronos-base | 102.3M | 0.14s | 1.68s | ~63.4 天 |

结论：CPU 推理可行 (30ms)，微调必须用 mini 或 GPU。

### 2.5 Kronos-mini 微调

- **20 只股票测试微调** ✅ 完成 (5 epochs)，模型保存至 `outputs/mini_finetune/final/`
- **3,592 只全量微调** 🔄 启动中 (30 epochs)
  - 训练样本：145,752
  - 验证样本：36,439
  - 批次大小：4

### 2.6 文档

| 文档 | 内容 |
|------|------|
| `CPU_TRAINING_REPORT.md` | 完整报告：工程概述、测试环境、三模型对比、FAQ (Q1-Q3)、技术深度问答 (Q4-Q7)、架构图 |
| `A_SHARE_DATA_REPORT.md` | A 股数据完整性分析、Kronos 微调适用性 |
| `SESSION_SUMMARY.md` | 本文件 |

### 2.7 Git 提交记录

```
3bfadcb — docs: add technical deep-dive (Q4-Q7)
afb8a57 — docs: update reports with 3-model comparison
c6b47ea — docs: clarify Q2 - fine-tuning is one-time cost
53e42c6 — docs: add A-share data completeness report
5a69ae9 — benchmark: compare Kronos mini/small/base CPU training time
c178465 — initial A-share adapter + CPU benchmark + report
```

---

## 三、技术问答 (Q1-Q7)

回答覆盖了以下主题：

1. **Kronos-mini vs small**：精度差距可控，CPU 训练时间急剧增加
2. **微调是一遍就够了**：one-time cost，产出模型后永久复用
3. **预训练 vs 微调**：预计改善 30-50% 预测误差
4. **成交量+价格作为 Token**：6 维 (O,H,L,C,V,Amount) 联合编码
5. **分钟级数据**：理论提升 15-25%，mini 的 2048 上下文有优势
6. **模型架构**：完整的 ASCII 架构图 (Tokenizer VQ-VAE → Predictor Autoregressive Transformer)
7. **增强 A 股适应性**：4 级路线图（预筛选 → 因子拼接 → 两阶段流水线 → 多模态融合）

---

## 四、待完成

| 任务 | 状态 |
|------|------|
| 全量 Kronos-mini 微调 (3,592 stocks, 30 epochs) | 🔄 进行中 |
| 微调后回测验证（历史某时点预测 vs 实际走势对比） | 📋 待做 |
| A 股数据更新至最新交易日 | 📋 待做 |
| 多因子增强实验（sector, market_cap 等） | 📋 待做 |

---

## 五、关键路径

```
Kronos/.venv/          ← Python 虚拟环境
Kronos/a_share_adapter/ ← 所有 A 股适配脚本
Kronos/outputs/        ← 模型 checkpoint + 预测图表
Kronos/*.md            ← 文档报告