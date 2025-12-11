# features/indicators.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_supertrend_and_cci(
    df_ohlc: pd.DataFrame,
    atr_period: int,
    multiplier: float,
    cci_period: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Общая реализация SuperTrend + CCI для произвольного OHLC-dataframe.

    Возвращает:
        st_series  – линия SuperTrend;
        dir_series – +1 / -1 (up / down);
        cci_series – CCI по typical price.
    """
    high = df_ohlc["high"].astype(float)
    low = df_ohlc["low"].astype(float)
    close = df_ohlc["close"].astype(float)

    # --- ATR для SuperTrend (локальный, независимый от "atr") ---
    prev_close = close.shift(1)
    high_low = high - low
    high_prev = (high - prev_close).abs()
    low_prev = (low - prev_close).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    atr_st = tr.rolling(window=atr_period, min_periods=1).mean()

    # --- Базовые линии upper / lower ---
    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr_st
    basic_lb = hl2 - multiplier * atr_st

    ub = basic_ub.to_numpy().copy()
    lb = basic_lb.to_numpy().copy()
    c = close.to_numpy()
    n = len(df_ohlc)

    # финальные ленты по классическому алгоритму
    for i in range(1, n):
        # верхняя лента
        if (basic_ub.iat[i] < ub[i - 1]) or (c[i - 1] > ub[i - 1]):
            ub[i] = basic_ub.iat[i]
        else:
            ub[i] = ub[i - 1]

        # нижняя лента
        if (basic_lb.iat[i] > lb[i - 1]) or (c[i - 1] < lb[i - 1]):
            lb[i] = basic_lb.iat[i]
        else:
            lb[i] = lb[i - 1]

    st = np.full(n, np.nan, dtype=float)
    direction = np.zeros(n, dtype=int)  # +1 = up, -1 = down

    # инициализация
    st[0] = ub[0]
    direction[0] = -1 if c[0] < ub[0] else 1

    for i in range(1, n):
        prev_st = st[i - 1]
        prev_ub = ub[i - 1]
        prev_lb = lb[i - 1]

        # предыдущий тренд вниз (линия = верхняя лента)
        if prev_st == prev_ub:
            if c[i] <= ub[i]:
                st[i] = ub[i]
                direction[i] = -1
            else:
                st[i] = lb[i]
                direction[i] = 1

        # предыдущий тренд вверх (линия = нижняя лента)
        elif prev_st == prev_lb:
            if c[i] >= lb[i]:
                st[i] = lb[i]
                direction[i] = 1
            else:
                st[i] = ub[i]
                direction[i] = -1
        else:
            # fallback на случай численных странностей
            st[i] = st[i - 1]
            direction[i] = direction[i - 1]

    st_series = pd.Series(st, index=df_ohlc.index)
    dir_series = pd.Series(direction, index=df_ohlc.index)

    # --- CCI по typical price ---
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=cci_period, min_periods=1).mean()
    mean_dev = (tp - sma_tp).abs().rolling(window=cci_period, min_periods=1).mean()
    denom = 0.015 * mean_dev.replace(0, np.nan)
    cci = ((tp - sma_tp) / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return st_series, dir_series, cci
