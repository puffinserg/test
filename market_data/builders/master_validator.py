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

    df = pd.read_parquet(path)
    print(f"[master_validator] Loaded {len(df)} rows from {path}")

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
    else:
        print("[master_validator] OK: no time gaps (uniform)")
