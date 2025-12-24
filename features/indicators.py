# features/indicators.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

def atr_wilder_mt(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    ATR максимально близко к MetaTrader iATR:
    - TR = max(H-L, |H-prevC|, |L-prevC|)
    - Wilder smoothing
    - первые period-1 значений = 0 (как "невалидные" в терминах индикатора)
    - первое валидное значение ATR ставим на индексе [period-1] как SMA(TR[0:period])
      (это важный сдвиг, который обычно даёт ровно такие расхождения как у тебя).
    """
    n = len(close)
    p = max(1, int(period))
    atr = np.zeros(n, dtype=float)
    if n == 0:
        return atr

    prev_close = np.empty(n, dtype=float)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])

    # первые (p-1) значений считаем "не готовыми"
    if n < p:
        return atr

    # MT-style старт: ATR на индексе (p-1) = SMA(TR[0:p])
    atr[p - 1] = float(np.mean(tr[0:p]))

    for t in range(p, n):
        atr[t] = (atr[t - 1] * (p - 1) + tr[t]) / p

    return atr

def compute_supertrend_and_cci(
    df_ohlc: pd.DataFrame,
    atr_period: int,
    cci_period: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high = df_ohlc["high"].astype(float).to_numpy()
    low  = df_ohlc["low"].astype(float).to_numpy()
    close= df_ohlc["close"].astype(float).to_numpy()
    n = len(df_ohlc)

    # CCI (без изменений)
    tp = (high + low + close) / 3.0
    p_cci = max(1, int(cci_period))
    tp_s = pd.Series(tp, index=df_ohlc.index)
    sma = tp_s.rolling(p_cci, min_periods=p_cci).mean()
    def _md(arr: np.ndarray) -> float:
        m = float(arr.mean())
        return float(np.mean(np.abs(arr - m)))
    md = tp_s.rolling(p_cci, min_periods=p_cci).apply(_md, raw=True)
    denom = (0.015 * md).replace(0, np.nan)
    cci_s = ((tp_s - sma) / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    cci = cci_s.to_numpy(dtype=float)

    # ATR (без изменений)
    atr = atr_wilder_mt(high, low, close, atr_period)

    # Buffers
    trend_up = np.zeros(n, dtype=float)
    trend_dn = np.zeros(n, dtype=float)
    st_line  = np.zeros(n, dtype=float)
    direction= np.zeros(n, dtype=int)

    st_threshold = 0.0

    for t in range(n):
        cci_now  = cci[t]
        cci_prev = cci[t-1] if t > 0 else 0.0

        if cci_now >= st_threshold:
            # Базовый уровень
            trend_up[t] = low[t] - atr[t]

            # Перенос ПЕРЕД липкостью, если смена
            if t > 0 and cci_prev < st_threshold:
                trend_up[t] = trend_dn[t-1]

            # Липкость (теперь применяется к возможно перенесённому уровню)
            if t > 0 and trend_up[t-1] != 0.0 and trend_up[t] < trend_up[t-1]:
                trend_up[t] = trend_up[t-1]

            st_line[t] = trend_up[t]
            direction[t] = 1
        else:
            trend_dn[t] = high[t] + atr[t]

            if t > 0 and cci_prev > st_threshold:
                trend_dn[t] = trend_up[t-1]

            if t > 0 and trend_dn[t-1] != 0.0 and trend_dn[t] > trend_dn[t-1]:
                trend_dn[t] = trend_dn[t-1]

            st_line[t] = trend_dn[t]
            direction[t] = -1

    st_series  = pd.Series(st_line, index=df_ohlc.index).astype(float)
    dir_series = pd.Series(direction, index=df_ohlc.index).astype(int)
    return st_series, dir_series, cci_s

def compute_atr(df_ohlc: pd.DataFrame, period: int) -> pd.Series:
    """
    ATR на основе стандартного True Range.
    Логика 1-в-1 как в feature_atr.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df_ohlc.columns):
        raise ValueError(f"compute_atr: OHLC columns missing: {required}")

    prev_close = df_ohlc["close"].shift(1)
    high_low = df_ohlc["high"] - df_ohlc["low"]
    high_prev = (df_ohlc["high"] - prev_close).abs()
    low_prev = (df_ohlc["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def compute_returns(
    close: pd.Series, short_periods: Sequence[int], long_period: int
) -> Dict[str, pd.Series]:
    """
    Доходности ret_p и ret_long_p по закрытию.
    Логика 1-в-1 как в feature_returns.
    """
    result: Dict[str, pd.Series] = {}

    for p in short_periods:
        col = f"ret_{p}"
        result[col] = close.pct_change(periods=p).fillna(0.0)

    col_long = f"ret_long_{long_period}"
    result[col_long] = close.pct_change(periods=long_period).fillna(0.0)

    return result


def compute_volatility(close: pd.Series, period: int) -> pd.Series:
    """
    Волатильность как rolling std от дневной доходности (ret_1).
    Логика 1-в-1 как в feature_volatility.
    """
    returns_1 = close.pct_change().fillna(0.0)
    vol = returns_1.rolling(window=period, min_periods=1).std().fillna(0.0)
    return vol


def compute_spread_stats(
    spread: pd.Series,
    period: int,
    atr: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Фичи по спреду:
      - среднее,
      - std,
      - отношение среднего спреда к ATR (spread_over_atr).

    Логика 1-в-1 как в feature_spread:
    - если ATR нет, spread_over_atr = 0.0.
    """
    spread_mean = spread.rolling(window=period, min_periods=1).mean().fillna(0.0)
    spread_std = spread.rolling(window=period, min_periods=1).std().fillna(0.0)

    if atr is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = spread_mean / atr
            ratio = ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        # точное соответствие старому коду: просто нули
        ratio = pd.Series(0.0, index=spread_mean.index)

    return spread_mean, spread_std, ratio

def compute_murrey_grid(
    df_ohlc: pd.DataFrame,
    period_bars: int,
    include_extremes: bool = True,
) -> dict[str, pd.Series]:
    """
    Классический Murrey-grid (упрощённая классика):
    1) Берём rolling HH/LL за period_bars
    2) Определяем "octave" как ближайшую степень двойки >= range
    3) Step = octave/8
    4) Base = floor(LL/step)*step
    5) Levels: base + i*step, i=0..8 (+ экстремальные -2..-1, +1..+2 если include_extremes)

    Возвращает словарь Series:
      levels: mur_-2_8..mur_10_8 (если include_extremes) иначе 0..8
      derived:
        mur_zone (0..7), mur_pos_in_zone (0..1),
        mur_nearest_idx (индекс уровня), mur_dist_close_to_nearest,
        mur_dist_close_to_0_8, mur_dist_close_to_4_8, mur_dist_close_to_8_8
    """
    required = {"high", "low", "close"}
    if not required.issubset(df_ohlc.columns):
        raise ValueError(f"compute_murrey_grid: OHLC columns missing: {required}")

    high = df_ohlc["high"].astype(float)
    low = df_ohlc["low"].astype(float)
    close = df_ohlc["close"].astype(float)

    hh = high.rolling(window=period_bars, min_periods=period_bars).max()
    ll = low.rolling(window=period_bars, min_periods=period_bars).min()

    rng = (hh - ll).abs()

    # octave = 2^ceil(log2(range)), но range может быть 0 или NaN
    eps = 1e-12
    safe_rng = rng.fillna(0.0).clip(lower=0.0) + eps
    octave = np.power(2.0, np.ceil(np.log2(safe_rng.to_numpy())))
    octave = pd.Series(octave, index=df_ohlc.index)

    step = octave / 8.0
    # base = floor(ll/step)*step
    base = (ll / step).apply(np.floor) * step

    # индексы уровней
    if include_extremes:
        idxs = list(range(-2, 11))  # -2..10 соответствует -2/8..10/8 (экстремы +2 уровня)
    else:
        idxs = list(range(0, 9))    # 0..8

    out: dict[str, pd.Series] = {}

    # уровни
    for i in idxs:
        out[f"mur_{i}_8"] = base + (i * step)

    # --- derived: зона и позиция ---
    # зона считается по базовому квадрату 0..8 (даже если экстремы включены)
    l0 = out["mur_0_8"]
    l8 = out["mur_8_8"]

    # step в теории одинаковый везде, но оставляем вычисленный
    step_safe = step.replace(0, np.nan)

    # зона: clamp 0..7
    zone_raw = ((close - l0) / step_safe).apply(np.floor)
    zone = zone_raw.clip(lower=0, upper=7).fillna(0).astype(int)

    # pos внутри зоны
    lvl_low = l0 + zone.astype(float) * step
    pos = ((close - lvl_low) / step_safe).clip(lower=0.0, upper=1.0).fillna(0.0)

    out["mur_zone"] = zone.astype(float)  # оставляем float, чтобы не ломать пайплайн
    out["mur_pos_in_zone"] = pos

    # --- derived: расстояния ---
    level_stack = pd.concat([out[f"mur_{i}_8"] for i in idxs], axis=1)

    # abs distance to each level
    dists = (level_stack.sub(close, axis=0)).abs()

    # IMPORTANT: avoid all-NA idxmin issues
    dists_safe = dists.fillna(np.inf)

    nearest_col = dists_safe.idxmin(axis=1)  # always a column label now
    dist_nearest = dists_safe.min(axis=1)  # inf if all were NaN

    # parse "mur_{i}_8" -> i
    nearest_idx = (
        nearest_col.astype("string")
        .str.extract(r"mur_(-?\d+)_8")[0]
        .astype(float)
        .fillna(0.0)
    )

    # replace inf (no data) with 0.0
    dist_nearest = dist_nearest.replace([np.inf], 0.0).fillna(0.0)

    out["mur_nearest_idx"] = nearest_idx
    out["mur_dist_close_to_nearest"] = dist_nearest

    # ключевые
    out["mur_dist_close_to_0_8"] = (close - out["mur_0_8"]).abs().fillna(0.0)
    out["mur_dist_close_to_4_8"] = (close - out["mur_4_8"]).abs().fillna(0.0)
    out["mur_dist_close_to_8_8"] = (close - out["mur_8_8"]).abs().fillna(0.0)

    return out

