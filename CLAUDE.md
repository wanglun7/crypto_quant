# Crypto‑Quant – Claude Code Project Memory

> **⚠ 本文件由人类维护，Claude 会在每次运行时自动读取。仅放置「长期不变」信息，短期讨论请写在 Issue / Pull Request。**

---

## 1 | 目标与 KPI

* **愿景** 构建可实盘的 BTC 永续合约量化系统（4h–tick）。
* **阶段 KPI**  (以回测结果为准)

  | 指标      | 目标       |
  | ------- | -------- |
  | 30 d 年化 | ≥ 70 %   |
  | Sharpe  | ≥ 1.5    |
  | 最大回撤    | ≤ 25 %   |
  | 执行延迟    | ≤ 100 ms |

---

## 2 | 目录约定

```
crypto_quant/
├── data_pipeline/       # 数据采集 & 预处理 (WS, REST)
│   ├── collectors/      # 单所采集器，实现 AsyncProducer 抽象
│   ├── storage/         # Timescale 写入工具
│   └── validators/      # 数据完整性检查
├── features/            # 增量特征计算
│   ├── microstructure/
│   ├── tech_indicators/
│   └── onchain/
├── models/              # 预测模型 (PyTorch)
├── strategy/            # 信号、仓位、风险
├── execution/           # 下单 & 路由
├── backtesting/         # 符合真实撮合的回测引擎
└── common/              # 公用 utils / config / enums
```

> **原则**：每个子目录自带 `__init__.py`, `README.md`, `tests/`。

---

## 3 | 技术栈

* **语言** Python 3.11 (主) + Rust (性能热点)
* **核心依赖**
  `asyncio`, `httpx`, `psycopg_pool`, `pandas`, `pytorch‑1.13+cpu`, `ta‑lib`, `vectorbt‑pro`, `ray[tune]`
* **基础设施**
  TimescaleDB ▶ Kafka ▶ Airflow / Prefect ▶ Prometheus ▶ Grafana

---

## 4 | Claude Code 工作流

| 步骤                   | 命令 / 规则                                                           |
| -------------------- | ----------------------------------------------------------------- |
| **① 选定作用域**          | `claude /add-dir <path>` — 仅添加当前子模块；严禁一次性全仓库。                     |
| **② 制定 PLAN**        | Prompt: *“Generate step‑by‑step PLAN for …, wait for ‘execute’.”* |
| **③ 生成 Patch**       | 确认后 `execute 1`；CLI 自动 `diff.tool auto` 生成补丁。                     |
| **④ 运行测试**           | `/run tests` (pytest + ruff + mypy)；红灯则 `rollback`.               |
| **⑤ Apply & Commit** | `/apply` → `git commit -m "feat: …"`.                             |
| **⑥ 结束会话**           | `/compact` every ≈ 40 turns; start new issue.                     |

### 常用 Slash

```
/plan        # 输出思路与分步动作
/review      # 审阅最近改动
/fix         # 让 Claude 修复失败测试
/add-dir     # 动态扩充上下文
/compact     # 折叠历史对话
```

---

## 5 | 代码风格

* **格式** `black --line-length 100`; import 排序用 `ruff ‑‑fix`。
* **类型检查** 强制 `mypy --strict`。
* **日志** 统一 `structlog`; 禁用裸 `print()`。
* **精度** 金融计算使用 `decimal.Decimal`, 严禁 float 链式相减。

---

## 6 | Sprint 模板

```md
### 🎯 目标
<一句话功能>

### ✅ 验收标准
- [ ] 单测通过 (`pytest`)
- [ ] 文档补全
- [ ] Prom 指标暴露

### 🔗 相关文件
<data_pipeline/collectors/binance_ws.py>
```

> 每个 GitHub Issue 必须用此模板；Issue URL 作为 Prompt 第一行给 Claude。

---

## 7 | TODO 里程碑

| Phase | Epic                           | Owner | 预计   | 状态 |
| ----- | ------------------------------ | ----- | ---- | -- |
| 1     | Binance WS Collector           | @lun  | 5 d  | 🚧 |
| 1     | Timescale Writer               | @lun  | 3 d  | ☐  |
| 2     | Feature – microstructure batch | @lun  | 7 d  | ☐  |
| 2     | Backtest engine (v1)           | @lun  | 10 d | ☐  |

>  更新请修改此表；Claude 阅读后能避免重复劳动。

---

**最后更新** :   <!-- KEEP THIS LINE FOR CLAUDE AUTO‑STAMP -->
