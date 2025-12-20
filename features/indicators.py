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
    SuperTrend + CCI в стиле MT5 (supertrendmt5.mq5):
      - CCI по PRICE_TYPICAL
      - ATR = Wilder (как iATR)
      - TrendUp = Low - ATR*mult, TrendDown = High + ATR*mult
      - выбор активной линии по знаку CCI
      - перенос буферов при смене знака CCI для непрерывности

    Возвращает:
      st_series  – единая линия SuperTrend (как "Trend" buffer),
      dir_series – +1/-1 по знаку CCI,
      cci_series – значение CCI.
    """
    high = df_ohlc["high"].astype(float)
    low = df_ohlc["low"].astype(float)
    close = df_ohlc["close"].astype(float)

    # ---------- True Range ----------
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # ---------- ATR Wilder (iATR) ----------
    tr_np = tr.to_numpy(dtype=float)
    n = len(tr_np)
    atr_np = np.full(n, np.nan, dtype=float)
    if n > 0:
        # старт: SMA(TR, period) на первом доступном окне
        p = max(1, int(atr_period))
        # для маленьких выборок не падаем
        first = min(p, n)
        atr_np[0] = np.nanmean(tr_np[:first])
        for i in range(1, n):
            atr_np[i] = (atr_np[i - 1] * (p - 1) + tr_np[i]) / p

    atr = pd.Series(atr_np, index=df_ohlc.index)

    # ---------- CCI PRICE_TYPICAL (как iCCI(..., PRICE_TYPICAL)) ----------
    tp = (high + low + close) / 3.0
    p = max(1, int(cci_period))
    sma_tp = tp.rolling(window=p, min_periods=p).mean()

    # mean deviation: mean(|tp_j - sma_tp_i|) по окну (важно: относительно текущего SMA)
    def _mean_dev(arr: np.ndarray) -> float:
        m = arr.mean()
        return float(np.mean(np.abs(arr - m)))

    md = tp.rolling(window=p, min_periods=p).apply(_mean_dev, raw=True)
    denom = 0.015 * md.replace(0, np.nan)
    cci = ((tp - sma_tp) / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ---------- MT5 SuperTrend logic (CCI-driven) ----------
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    cci_np = cci.to_numpy(dtype=float)
    atrv = atr.to_numpy(dtype=float) * float(multiplier)

    trend_up = np.full(n, np.nan, dtype=float)
    trend_dn = np.full(n, np.nan, dtype=float)
    trend = np.full(n, np.nan, dtype=float)
    direction = np.full(n, 0, dtype=int)

    for i in range(n):
        # базовые уровни
        tu = l[i] - atrv[i]
        td = h[i] + atrv[i]

        if i == 0:
            trend_up[i] = tu
            trend_dn[i] = td
        else:
            # перенос при смене знака (как в mq5)
            if cci_np[i] >= 0 and cci_np[i - 1] < 0:
                trend_up[i - 1] = trend_dn[i - 1]
            if cci_np[i] <= 0 and cci_np[i - 1] > 0:
                trend_dn[i - 1] = trend_up[i - 1]

            # "подтягивание" уровней
            if c[i - 1] > trend_up[i - 1]:
                tu = max(tu, trend_up[i - 1])
            if c[i - 1] < trend_dn[i - 1]:
                td = min(td, trend_dn[i - 1])

            trend_up[i] = tu
            trend_dn[i] = td

        if cci_np[i] >= 0:
            trend[i] = trend_up[i]
            direction[i] = 1
        else:
            trend[i] = trend_dn[i]
            direction[i] = -1

    st_series = pd.Series(trend, index=df_ohlc.index)
    dir_series = pd.Series(direction, index=df_ohlc.index)
    return st_series, dir_series, cci

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

