"""
Replay the walk-forward evaluation from the FROZEN committed sample only.

This is the actual reproducibility check: it does NOT touch the network or
data_manager_worldclass.py / feature_engine_worldclass.py at all. It loads
data/processed/sample_features_5m.parquet (the exact feature matrix
scripts/run_real_pipeline.py produced and committed) and re-runs the same
walk-forward LightGBM training + evaluation + cost-aware backtest on it.

Anyone who clones this repo can run `python scripts/replay_from_sample.py`
with no Binance API access and no re-fetching, and get the same numbers
that are in results/metrics.csv (bar floating-point/LightGBM-version noise).
It also reports the backtest result WITHOUT transaction costs, for direct
comparison against the cost-aware number in results/metrics.csv.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score  # noqa: E402

SEED = 42
np.random.seed(SEED)

TARGET_HORIZON = 5
TRAIN_DAYS, TEST_DAYS, STEP_DAYS, EMBARGO_MINUTES = 20, 5, 5, 60
TAKER_FEE, SLIPPAGE = 0.0010, 0.0005
ROUND_TRIP_COST = 2 * (TAKER_FEE + SLIPPAGE)
CONFIDENCE_THRESHOLD = 0.6

SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "sample_features_5m.parquet"


def make_windows(index):
    start, end = index.min(), index.max()
    windows, cur = [], start
    while cur + pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS) <= end:
        train_end = cur + pd.Timedelta(days=TRAIN_DAYS)
        test_start = train_end + pd.Timedelta(minutes=EMBARGO_MINUTES)
        test_end = min(test_start + pd.Timedelta(days=TEST_DAYS), end)
        windows.append((cur, train_end, test_start, test_end))
        cur += pd.Timedelta(days=STEP_DAYS)
    return windows


def backtest(y_true, proba, prices, apply_costs: bool):
    signal = np.where(proba > CONFIDENCE_THRESHOLD, 1,
              np.where(proba < 1 - CONFIDENCE_THRESHOLD, -1, 0))
    prices = prices.values
    trades, pos, entry = [], 0, None
    cost = ROUND_TRIP_COST if apply_costs else 0.0
    for i in range(len(signal)):
        if signal[i] != 0 and signal[i] != pos:
            if pos != 0 and entry is not None:
                trades.append(pos * (prices[i] - entry) / entry - cost)
            entry, pos = prices[i], signal[i]
    if pos != 0 and entry is not None:
        trades.append(pos * (prices[-1] - entry) / entry - cost)
    trades = np.array(trades) if trades else np.array([])
    total_return = float(np.prod(1 + trades) - 1) if len(trades) else 0.0
    win_rate = float((trades > 0).mean()) if len(trades) else 0.0
    return {"num_trades": int(len(trades)), "total_return": total_return, "win_rate": win_rate}


def main():
    if not SAMPLE_PATH.exists():
        print(f"ERROR: {SAMPLE_PATH} not found. This script replays the committed sample; "
              f"run scripts/run_real_pipeline.py first if you want to re-fetch fresh data.")
        sys.exit(1)

    df = pd.read_parquet(SAMPLE_PATH)
    target_col = f"target_{TARGET_HORIZON}m"
    feature_cols = [c for c in df.columns if not c.startswith("target_") and not c.startswith("forward_return_")]
    print(f"Loaded {SAMPLE_PATH.relative_to(REPO_ROOT)}: {len(df)} rows x {len(df.columns)} cols "
          f"({df.index.min()} .. {df.index.max()})")

    windows = make_windows(df.index)
    print(f"Walk-forward windows: {len(windows)}")

    oof_y, oof_proba, oof_price = [], [], []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, start=1):
        train_mask = (df.index >= tr_s) & (df.index < tr_e)
        test_mask = (df.index >= te_s) & (df.index < te_e)
        df_train, df_test = df[train_mask], df[test_mask]
        if len(df_train) < 200 or len(df_test) < 20:
            print(f"Fold {i}: insufficient rows, skipping")
            continue

        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbose=-1,
        )
        model.fit(df_train[feature_cols].values, df_train[target_col].values)
        proba = model.predict_proba(df_test[feature_cols].values)[:, 1]
        y_test = df_test[target_col].values
        pred = (proba > 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else float("nan")
        print(f"Fold {i}: n_train={len(df_train)} n_test={len(df_test)} acc={acc:.4f} auroc={auc:.4f}")

        oof_y.append(y_test)
        oof_proba.append(proba)
        oof_price.append(df_test["close"])

    y_all = np.concatenate(oof_y)
    proba_all = np.concatenate(oof_proba)
    pred_all = (proba_all > 0.5).astype(int)
    price_all = pd.concat(oof_price)

    print("\n=== Pooled out-of-fold (replayed from committed sample) ===")
    print(f"Accuracy: {accuracy_score(y_all, pred_all):.4f}")
    print(f"AUROC:    {roc_auc_score(y_all, proba_all):.4f}")
    print(f"F1:       {f1_score(y_all, pred_all):.4f}")

    bt_cost = backtest(y_all, proba_all, price_all, apply_costs=True)
    bt_raw = backtest(y_all, proba_all, price_all, apply_costs=False)
    print(f"\nBacktest WITHOUT costs: {bt_raw['num_trades']} trades, "
          f"return = {bt_raw['total_return']:+.4%}, win rate = {bt_raw['win_rate']:.2%}")
    print(f"Backtest WITH costs ({ROUND_TRIP_COST:.2%} round-trip): {bt_cost['num_trades']} trades, "
          f"return = {bt_cost['total_return']:+.4%}, win rate = {bt_cost['win_rate']:.2%}")
    print(f"Cost impact: {(bt_cost['total_return'] - bt_raw['total_return']) * 100:+.2f} pp")

    print("\n=== Confidence sweep (real, pooled OOF) ===")
    conf = np.maximum(proba_all, 1 - proba_all)
    print(f"{'threshold':>10} {'coverage':>10} {'accuracy':>10} {'n':>8}")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        m = conf >= thresh
        if m.sum() == 0:
            continue
        cov = m.mean()
        acc_at = accuracy_score(y_all[m], pred_all[m])
        print(f"{thresh:>10.2f} {cov:>10.1%} {acc_at:>10.4f} {int(m.sum()):>8}")


if __name__ == "__main__":
    main()
