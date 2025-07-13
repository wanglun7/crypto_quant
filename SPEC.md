# SPEC: fetch_ohlcv

目标：在给定交易所 (binance)、交易对、时间范围、时间粒度（如 "1m", "5m"）的情况下，
返回一张 **行数完整、时间索引单调递增且无缺口** 的 OHLCV DataFrame。

## 业务规则
1. 必须使用 `ccxt.binance().fetch_ohlcv` 获取原始数据；如超出 API 限制，可递归分页。
2. 索引必须为 `pd.DatetimeIndex`（UTC），并精确到毫秒。
3. 字段顺序固定：`["open","high","low","close","volume"]`。
4. 任意字段出现缺失值或时间缺口时，应 **raise ValueError**。
5. 结果 **不得直接写硬盘**（cache 单独处理）。

# SPEC: TA Multi-Scale Features

目标：在已有 OHLCV DataFrame (freq=1m) 的基础上，平滑计算多尺度技术指标：

| 分类 | 指标名 (列名) | 公式 / 备注 | 窗口 |
|------|--------------|-------------|------|
| 趋势 | EMA_fast (ema_fast), EMA_slow (ema_slow) | `ta.ema(close)` | 12, 26 |
| 动量 | MACD (macd), MACD_signal (macd_signal) | 12-26-9-EMA 组合 | – |
| 震荡 | RSI (rsi) | 14 | – |
| 波动 | BB_upper / BB_lower / BB_width | σ=20；宽度 = (U-L)/M | 20 |
| 强度 | ADX (adx) | 14 | – |

业务规则  
1. 传入 `df_1m`，index 为 UTC；函数 `generate_features(df_1m, scales=['1m','5m'])` 返回合并后特征表。  
2. 对每个 {scale, indicator} 生成列名 `<indicator>_<scale>`。  
3. 所有列 **不得出现 NaN**；对窗口前期不足处用前向填充后再整体 `.dropna()`。  
4. 输出 index 必须与输入 1m df *完全一致*（行数相等、对齐）。  
5. 若 `scales` 中存在非 1m 粒度，则先重采样（OHLC/Last/Mean），再计算指标，然后向 1m 向下填充对齐。  
6. 遇到未知指标/scale 时必须 `raise ValueError`。  

# SPEC: Baseline EMA Strategy

## 目标
实现 `run_backtest(df_ohlcv: pd.DataFrame, df_feat: pd.DataFrame, *, fee_maker=0.0001, fee_taker=0.0003)`  
返回 `stats: dict` 至少包含：
| key | 说明 |
|-----|------|
| total_return | (最后净值 - 1) |
| sharpe | 日频 Sharpe Ratio |
| max_dd | Max Drawdown |
| trade_count | 交易次数 |

## 信号逻辑
- 使用 1m 特征列：`ema_fast_1m`, `ema_slow_1m`
- `pos =  1` 若 `ema_fast > ema_slow`  
  `pos = -1` 若 `ema_fast < ema_slow`  
  `pos =  0` 若两者相等
- 每根 1 m 收盘换仓，按 **taker 费率** 计算手续费；仓位持有期间按 **maker 费率** 计滚动费用。

## 回测要求
1. 必须使用 **vectorbt v0.25+** 的 `Portfolio.from_signals` 或等价实现，保证秒速回测。
2. 时间索引与 `df_ohlcv` 完全一致，滑点 = 0。
3. 测试段（两小时样例）指标阈值：  
   - `sharpe ≥ 0.80`  
   - `max_dd ≤ 0.25`
4. 函数内部禁止 `print()`；应返回字典，由测试断言。
5. 若输入列缺失应 `raise ValueError`.

# SPEC · GRU Signal Pipeline  (vA-2025-07-14)

## 数据准备
1. 输入  
   - `df_ohlcv` (1-min, columns: open/high/low/close/volume)  
   - `df_feat` (由 ta_multi_scale 生成，多尺度特征)
2. **lookback = 60 根**；**horizon = 5 根**  
3. 标签 `y ∈ {0,1,2}`  
fut_ret = log(close_t+horizon / close_t)
if fut_ret > +0.0003 → y = 2 # Up (+0.03%)
elif fut_ret < −0.0003 → y = 0 # Down (−0.03%)
else → y = 1 # Hold

4. 数据拆分：按时间顺序  
- Train 70 %  
- Val   15 %  
- Test  15 %

## 模型
- **结构**：3 层 GRU(hidden = 32, dropout 0.2) → FC(3) → Softmax  
- **损失**：CrossEntropy(weight = class_freq^-1)  
- **优化**：Adam(lr 1e-3) + ReduceLROnPlateau(patience 1, factor 0.5)  
- **训练**：最多 10 epoch；EarlyStopping (val_bacc, patience 3)

## 推理 → 仓位
start_ix = lookback + horizon # 对齐去前瞻
if p_up ≥ 0.45 → pos = +1
elif p_down ≥0.45→ pos = -1
else pos = 0


## 回测要求
| 指标 | 阈值 |
|------|------|
| val_bacc | ≥ 0.52 |
| test_bacc | ≥ 0.50 |
| sharpe (30 d) | ≥ Baseline Sharpe + 0.10 |
| max_dd | ≤ 0.25 |
| lookahead | 不得使用未来特征/价格 |

### 交易参数  
- maker_fee = 0.0002   # 2 bp / min 持仓费  
- taker_fee = 0.0004   # 4 bp / 调仓  
- slippage  = 0.0002   # 2 bp
- 调仓方式：仅在信号翻转时下单，使用 `vectorbt.Portfolio.from_signals`
