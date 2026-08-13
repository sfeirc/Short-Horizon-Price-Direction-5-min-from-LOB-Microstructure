# Transaction Cost Model

> **2026-08-13 audit note:** the specific numbers in "Cost Impact on Performance" below were found
> to be hardcoded in `generate_results_plots.py`, not measured. A real run on 2026-08-13 measured
> `+66.45%` (before costs) → `-7.76%` (after costs) on 197 real walk-forward trades — costs flipped
> the sign, not just shrank it. See the main [README's Reproducibility section](../README.md#-reproducibility-audit-2026-08-13)
> for full details and caveats (the underlying trade tape is a filtered, non-contiguous timeline —
> treat these as illustrative, not validated returns). The numbers below are the ORIGINAL,
> unverified claim, kept for historical reference only.

## Overview

All backtest results include transaction costs:
- **Fees:** 0.1% per trade (Binance spot trading fee)
- **Slippage:** 0.05% average (estimated from order book depth)

**Rationale:** Including costs is essential for realistic performance evaluation.

## Fee Model

### Binance Spot Trading Fees

**Assumption:** 0.1% per trade (maker/taker)

**Implementation:**
```python
fee = trade_value * 0.001  # 0.1%
pnl_after_fees = pnl - fee
```

**Note:** Assumes no fee discounts (VIP levels). Real fees may be lower for high-volume traders.

## Slippage Model

### Order Book Depth Analysis

**Method:**
1. For each trade, estimate market impact from order book depth
2. Use top-N levels to compute average execution price
3. Slippage = `|execution_price - mid_price| / mid_price`

**Simplified Model:**
- Average slippage: **0.05%** (conservative estimate)
- Assumes small order sizes relative to book depth

**Real-World Considerations:**
- Actual slippage depends on:
  - Order size vs. market depth
  - Market volatility (higher vol → higher slippage)
  - Time-of-day (lower liquidity → higher slippage)
- Our model uses a fixed 0.05% average (future: dynamic slippage based on volatility/liquidity)

## Cost Impact on Performance

### Without Costs (Unrealistic)
- Accuracy: `84.2%`
- PnL: `+18.5%` (30-day)
- Sharpe: `3.1`

### With Costs (Realistic)
- Accuracy: `84.2%` (unchanged)
- PnL: `+14.2%` (30-day) ← **–4.3 pp** impact
- Sharpe: `2.8` ← **–0.3** impact

**Conclusion:** Costs reduce returns by ~23%, but model remains profitable.

## Future Improvements

1. **Dynamic Slippage:**
   - Volatility-dependent: `slippage = base_slippage * (1 + vol_multiplier)`
   - Liquidity-dependent: `slippage = f(order_size, book_depth)`

2. **Latency Costs:**
   - Model for execution latency (ms-level)
   - Queue position modeling

3. **Spread Costs:**
   - Bid-ask spread as cost (not just slippage)

4. **Real Exchange Data:**
   - Use historical execution logs for realistic cost estimates

