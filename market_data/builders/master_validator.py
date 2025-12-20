# RL_PyTorch/market_data/builders/master_validator.py
from __future__ import annotations

import pandas as pd

from config.paths import MASTER_DIR
from config.settings import SETTINGS

from .master_builder import _master_file_path


def validate_master_history_impl() -> None:
    path = _master_file_path()
    if not path.exists():
        print(f"[master_validator] Master file not found: {path}")
        return

    # читаем только нужные колонки (быстрее и меньше памяти)
    cols = ["time", "open", "high", "low", "close"]
    df = pd.read_parquet(path, columns=[c for c in cols if c])
    print(f"[master_validator] Loaded {len(df)} rows from {path}")

    if "time" not in df.columns:
        print("[master_validator] ERROR: 'time' column not found!")
        return

    # time -> datetime (UTC)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # сортировка
    if not df["time"].is_monotonic_increasing:
        print("[master_validator] WARNING: 'time' is not sorted, sorting...")
        df = df.sort_values("time").reset_index(drop=True)

    dup_count = df["time"].duplicated().sum()
    if dup_count > 0:
        print(f"[master_validator] WARNING: {dup_count} duplicated timestamps found")
    else:
        print("[master_validator] OK: no duplicated timestamps")

    diffs = df["time"].diff().dropna()
    # Самый частый шаг (часто 1 минута, но могут быть дырки)
    main_step = diffs.mode()[0]
    gaps = diffs[diffs != main_step]
    if not gaps.empty:
        print(f"[master_validator] WARNING: {len(gaps)} irregular time gaps found")
        print("  examples:")
        print(gaps.head())
        # полезная статистика
        try:
            print(f"[master_validator] main_step={main_step}, min_gap={diffs.min()}, max_gap={diffs.max()}")
        except Exception:
            pass
    else:
        print("[master_validator] OK: no time gaps (uniform)")

    # ---- OHLC sanity checks ----
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[master_validator] WARNING: missing OHLC columns: {missing}")
        return

    # NaN check
    nan_any = df[required].isna().any().any()
    if nan_any:
        nan_rows = df[df[required].isna().any(axis=1)]
        print(f"[master_validator] WARNING: NaNs in OHLC: rows={len(nan_rows)}")
        print("[master_validator] First NaN rows:")
        print(nan_rows.head(3))
        print("[master_validator] Last NaN rows:")
        print(nan_rows.tail(3))
    else:
        print("[master_validator] OK: no NaNs in OHLC columns")

    # Zero OHLC check (это критично — 0 цен быть не должно)
    zero_mask = (df[required] == 0).any(axis=1)
    zero_cnt = int(zero_mask.sum())
    if zero_cnt > 0:
        z = df.loc[zero_mask, ["time"] + required]
        print(f"[master_validator] ERROR: ZERO OHLC detected: rows={zero_cnt}")
        print("[master_validator] First zero rows:")
        print(z.head(5))
        print("[master_validator] Last zero rows:")
        print(z.tail(5))
    else:
        print("[master_validator] OK: no ZERO values in OHLC")

    # Candle consistency (high/low must bound open/close)
    bad_hl = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1)) | (df["high"] < df["low"])
    bad_cnt = int(bad_hl.sum())
    if bad_cnt > 0:
        b = df.loc[bad_hl, ["time"] + required]
        print(f"[master_validator] WARNING: inconsistent candles: rows={bad_cnt}")
        print("[master_validator] First bad candles:")
        print(b.head(5))
        print("[master_validator] Last bad candles:")
        print(b.tail(5))
    else:
        print("[master_validator] OK: candle bounds look consistent")
