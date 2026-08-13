# Evaluation Methodology

> **2026-08-13 audit note:** the specific numbers throughout this document (84.2% accuracy, ablation
> deltas, 82.3% ETH accuracy, precision-at-confidence figures) were found to be hardcoded in
> `generate_results_plots.py`, never measured. A real run on 2026-08-13 measured AUROC 0.922 /
> Accuracy 85.0% (pooled walk-forward out-of-fold, LightGBM only); ablations and the cross-asset
> ETH test were not reproduced. See the main
> [README's Reproducibility section](../README.md#-reproducibility-audit-2026-08-13) for the real
> numbers, what was and wasn't reproduced, and how to re-run it yourself. The methodology
> *description* below (walk-forward mechanics, regime bucketing, label rules) remains an accurate
> description of what the code does; only the specific reported numbers are unverified/historical.

## Temporal Validation

### Walk-Forward Time-Series Cross-Validation

To avoid data leakage, we use strict temporal splitting:

```
Training Window: [T-90d, T]
Test Window:     [T, T+7d]
Embargo:         1 day (no overlap)

Step forward:    T → T+7d
Repeat until:   T+7d reaches end of data
```

**Implementation:**
- Expanding window: Training set grows over time
- Rolling option: Fixed 90-day window (future extension)
- Embargo period: 1 day gap prevents leakage from test set features

### Label Latency Enforcement

Target labels computed from future prices:
- **Prediction time:** `t`
- **Target return:** `(close[t+5] - close[t]) / close[t]`
- **Label:** `1` if return > `+0.2%`, `0` if return < `-0.2%`, else dropped (this is the actual value
  in `config_ultimate.yaml`; this doc previously said ±0.15%)

**Guarantee:** Features at time `t` never use information from `t+1` onwards.

### Neutral Zone Filtering

To reduce label noise, we filter "neutral" samples:
- Drop samples where `|return| < 0.2%` **on the 5-, 10-, AND 15-minute horizon simultaneously**
  (a row needs a decisive label on all three to survive `_clean_data()`'s `dropna()`)
- Rationale: Small moves (< 0.2%) are dominated by noise, not signal

**Impact:** measured 2026-08-13 at **87.3%** of rows dropped (8,203 kept of 64,801 raw 1-minute
bars) — not the "~30%" previously estimated here. See the main README's Reproducibility section for
what this means for how the reported accuracy/AUROC should be interpreted.

## Regime-Aware Evaluation

### Volatility Regime Buckets

**Method:**
1. Compute 24-hour realized volatility (rolling window)
2. Calculate percentile ranks: `vol_percentile = rolling_vol.quantile(0.5) / rolling_vol.max()`
3. Bucket assignment:
   - **Low Vol:** `< 33rd percentile`
   - **Mid Vol:** `33rd - 67th percentile`
   - **High Vol:** `> 67th percentile`

**Rationale:** Model performance varies by market conditions. Reporting per-regime metrics ensures robustness.

### Per-Regime Metrics

For each regime bucket:
- Accuracy
- AUROC
- Sharpe Ratio
- Maximum Drawdown
- Precision @ confidence ≥ 0.6

## Ablation Studies

### Feature Family Ablation

To identify which features contribute most to performance:

1. **Remove feature family** (e.g., order-book imbalance)
2. **Retrain model** on reduced feature set
3. **Evaluate** on same test set
4. **Compute delta:** `Δ = metric(full) - metric(ablated)`

**Reported deltas:**
- Order-book imbalance: `–3.2 pp accuracy` (high-vol)
- Time-of-day: `–1.1 pp`
- Multi-timeframe returns: `–2.8 pp`
- Volatility features: `–2.1 pp`
- Cross-asset: `–1.5 pp`
- Technical indicators: `–0.8 pp`

### Conclusion

Order-book microstructure features provide the largest incremental signal, followed by multi-timeframe returns.

## Cross-Asset Generalization

### Train on BTC, Validate on ETH

**Setup:**
- Training: BTCUSDT only
- Validation: ETHUSDT only

**Rationale:** Tests model generalization to different assets with similar structure.

**Result:** 82.3% accuracy on ETH (vs 84.2% on BTC), demonstrating reasonable generalization.

## Confidence Filtering

### Precision vs Confidence Threshold

**Method:**
1. Predict probabilities for all test samples
2. Filter by confidence threshold: `confidence = max(p, 1-p)`
3. Report precision on filtered subset

**Observation:** Precision increases with confidence threshold:
- All predictions: `84.2%` accuracy
- Confidence ≥ 0.6: `87.5%` accuracy
- Confidence ≥ 0.7: `90.1%` accuracy

**Implication:** Model is well-calibrated; high-confidence predictions are more reliable.

## Reproducibility Guarantees

### Fixed Seeds
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- scikit-learn: `random_state=42`

### Config-Driven
All hyperparameters in `config_ultimate.yaml`:
- Model parameters
- Feature toggles
- Validation parameters
- Cost assumptions

### Cached Data
Raw data cached to disk (Parquet format):
- Avoids repeated API calls
- Ensures consistent data across runs

### CI Validation
GitHub Actions run smoke tests:
- Validate metrics don't regress
- Check code integrity

