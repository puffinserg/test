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
    """
    SUPER-TREND 1:1 как в supertrendmt5.mq5:
      - CCI: PRICE_TYPICAL, period = CCI_Period
      - ATR: iATR (Wilder), period = ATR_Period, БЕЗ multiplier
      - TrendUp = Low - ATR
      - TrendDown = High + ATR
      - перенос при смене знака CCI: TrendUp[t-1] = TrendDown[t-1] / наоборот
      - "липкость" уровней:
          if TrendUp[t-1]!=0 and TrendUp[t] < TrendUp[t-1] => TrendUp[t]=TrendUp[t-1]
          if TrendDown[t-1]!=0 and TrendDown[t] > TrendDown[t-1] => TrendDown[t]=TrendDown[t-1]
    Возвращаем одну линию st (как st0 из MT4-экспорта), dir и cci.
    """
    high = df_ohlc["high"].astype(float).to_numpy()
    low  = df_ohlc["low"].astype(float).to_numpy()
    close= df_ohlc["close"].astype(float).to_numpy()
    n = len(df_ohlc)

    # -------- CCI PRICE_TYPICAL --------
    tp = (high + low + close) / 3.0
    p_cci = max(1, int(cci_period))

    tp_s = pd.Series(tp, index=df_ohlc.index)
    sma = tp_s.rolling(p_cci, min_periods=p_cci).mean()

    # mean deviation относительно SMA текущего окна (это и есть среднее)
    def _md(arr: np.ndarray) -> float:
        m = float(arr.mean())
        return float(np.mean(np.abs(arr - m)))

    md = tp_s.rolling(p_cci, min_periods=p_cci).apply(_md, raw=True)
    denom = (0.015 * md).replace(0, np.nan)
    cci_s = ((tp_s - sma) / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    cci = cci_s.to_numpy(dtype=float)

    # -------- TR --------
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ])

    # -------- ATR как iATR (MT-style) --------
    atr = atr_wilder_mt(high, low, close, atr_period)

    # -------- SuperTrend buffers (как в MQ5: 0 = empty) --------
    trend_up = np.zeros(n, dtype=float)
    trend_dn = np.zeros(n, dtype=float)
    st_line  = np.zeros(n, dtype=float)   # итоговая линия (как st0)
    direction= np.zeros(n, dtype=int)

    st_threshold = 0.0  # как double st=0.0 в mq5

    for t in range(n):
        cci_now  = cci[t]
        cci_prev = cci[t-1] if t > 0 else cci[t]

        # перенос на ПРЕДЫДУЩЕМ баре (как TrendUp[i+1] = TrendDown[i+1])
        if t > 0:
            if cci_now >= st_threshold and cci_prev < st_threshold:
                trend_up[t-1] = trend_dn[t-1]
            if cci_now <= st_threshold and cci_prev > st_threshold:
                trend_dn[t-1] = trend_up[t-1]
        if cci_now >= st_threshold:
            # MT5 индикатор: Low - ATR (без коэффициента)
            trend_up[t] = low[t] - atr[t]

            if t > 0 and trend_up[t-1] != 0.0 and trend_up[t] < trend_up[t-1]:
                trend_up[t] = trend_up[t-1]
            st_line[t] = trend_up[t]
            direction[t] = 1
        else:
            # MT5 индикатор: High + ATR (без коэффициента)
            trend_dn[t] = high[t] + atr[t]
            if t > 0 and trend_dn[t-1] != 0.0 and trend_dn[t] > trend_dn[t-1]:
                trend_dn[t] = trend_dn[t-1]
            st_line[t] = trend_dn[t]
            direction[t] = -1

    # Чтобы поведение совпало с твоим CSV (где пустоты не нужны) — отдаём одну линию st_line
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

class MurreyMT4State:
    """
    Хранит состояние для эмуляции MT4-поведения (кэширование уровней).
    Один объект на символ/TF.
    """
    def __init__(self):
        self.last_time = None
        self.cached_levels = None  # dict с уровнями

def compute_murrey_grid_mt4_exact(df: pd.DataFrame,
                                  period_bars: int = 64,
                                  include_extremes: bool = True,
                                  state: MurreyMT4State = None) -> dict:
    """
    Точная копия Murrey Math VG из MT4.
    - Пересчёт только при новом баре (эмуляция nTime != Time[0]).
    - Кэширование уровней до смены бара.
    df должен быть отсортирован по времени по возрастанию, с колонками open, high, low, close.
    """
    if state is None:
        state = MurreyMT4State()

    # Текущий бар (последний)
    current_time = df.index[-1]

    print("Murrey DEBUF: compute_murrey_grid_mt4_exact")

    # Если бар не изменился — возвращаем кэшированные уровни
    if state.last_time == current_time and state.cached_levels is not None:
        return state.cached_levels

    # Новый бар — пересчитываем
    if len(df) < period_bars:
        levels = {f"mur_{i}_8": np.nan for i in range(-2, 11)}
        levels.update({
            "mur_zone": np.nan,
            "mur_pos_in_zone": np.nan,
            "mur_nearest_idx": np.nan,
            "mur_dist_close_to_nearest": np.nan,
            "mur_dist_close_to_0_8": np.nan,
            "mur_dist_close_to_4_8": np.nan,
            "mur_dist_close_to_8_8": np.nan,
        })
        state.cached_levels = levels
        state.last_time = current_time
        return levels

    high = df["high"].values[-period_bars:]
    low = df["low"].values[-period_bars:]
    close = df["close"].iloc[-1]

    v1 = np.min(low)
    v2 = np.max(high)

    # Fractal selection (точно как в MT4)
    if 25000 < v2 <= 250000:
        fractal = 100000
    elif 2500 < v2 <= 25000:
        fractal = 10000
    elif 250 < v2 <= 2500:
        fractal = 1000
    elif 25 < v2 <= 250:
        fractal = 100
    elif 12.5 < v2 <= 25:
        fractal = 12.5
    elif 6.25 < v2 <= 12.5:
        fractal = 12.5
    elif 3.125 < v2 <= 6.25:
        fractal = 6.25
    elif 1.5625 < v2 <= 3.125:
        fractal = 3.125
    elif 0.390625 < v2 <= 1.5625:
        fractal = 1.5625
    elif v2 > 0:
        fractal = 0.1953125
    else:
        fractal = 1.0

    range_val = v2 - v1
    if range_val == 0:
        range_val = 1e-10

    sum_val = np.floor(np.log(fractal / range_val) / np.log(2))
    octave = fractal * (0.5 ** sum_val)

    mn = np.floor(v1 / octave) * octave
    mx = mn + octave if (mn + octave) > v2 else mn + 2 * octave

    diff = mx - mn

    # x1–x6
    x1 = x2 = x3 = x4 = x5 = x6 = 0.0

    if (v1 >= (3*diff/16 + mn)) and (v2 <= (9*diff/16 + mn)):
        x2 = mn + diff/2
    if (v1 >= (mn - diff/8)) and (v2 <= (5*diff/8 + mn)) and x2 == 0:
        x1 = mn + diff/2
    if (v1 >= (mn + 7*diff/16)) and (v2 <= (13*diff/16 + mn)):
        x4 = mn + 3*diff/4
    if (v1 >= (mn + 3*diff/8)) and (v2 <= (9*diff/8 + mn)) and x4 == 0:
        x5 = mx
    if (v1 >= (mn + diff/8)) and (v2 <= (7*diff/8 + mn)) and (x1 + x2 + x4 + x5 == 0):
        x3 = mn + 3*diff/4
    if x1 + x2 + x3 + x4 + x5 == 0:
        x6 = mx

    final_h = x1 + x2 + x3 + x4 + x5 + x6

    # y1–y6
    y1 = mn if x1 > 0 else 0.0
    y2 = mn + diff/4 if x2 > 0 else 0.0
    y3 = mn + diff/4 if x3 > 0 else 0.0
    y4 = mn + diff/2 if x4 > 0 else 0.0
    y5 = mn + diff/2 if x5 > 0 else 0.0
    y6 = mn if (final_h > 0 and (y1 + y2 + y3 + y4 + y5 == 0)) else 0.0

    final_l = y1 + y2 + y3 + y4 + y5 + y6

    dmml = (final_h - final_l) / 8.0 if (final_h - final_l) != 0 else 1e-10

    # Уровни
    levels = {}
    levels["mur_-2_8"] = final_l - 2 * dmml
    levels["mur_-1_8"] = final_l - dmml
    for i in range(9):
        levels[f"mur_{i}_8"] = final_l + i * dmml
    levels["mur_9_8"] = final_h + dmml
    levels["mur_10_8"] = final_h + 2 * dmml

    # Derived фичи (как в твоей текущей версии)
    l0 = levels["mur_0_8"]
    l8 = levels["mur_8_8"]

    if close < l0:
        zone = -1
    elif close < l8:
        zone = (close - l0) / (l8 - l0) * 8  # 0..8
    else:
        zone = 9

    # pos_in_zone (0..1 внутри текущей 1/8)
    all_levels = [levels[f"mur_{i}_8"] for i in range(-2, 11)]
    nearest = min(all_levels, key=lambda x: abs(x - close))
    nearest_idx = all_levels.index(nearest)

    pos = (close - nearest) / dmml if dmml != 0 else 0.0

    levels["mur_zone"] = zone
    levels["mur_pos_in_zone"] = pos
    levels["mur_nearest_idx"] = nearest_idx
    levels["mur_dist_close_to_nearest"] = abs(close - nearest)
    levels["mur_dist_close_to_0_8"] = abs(close - l0)
    levels["mur_dist_close_to_4_8"] = abs(close - levels["mur_4_8"])
    levels["mur_dist_close_to_8_8"] = abs(close - l8)

    # Кэшируем
    state.cached_levels = levels
    state.last_time = current_time

    return levels

def compute_murrey_grid_mt4_vg(df: pd.DataFrame, period_bars: int = 64, include_extremes: bool = True):
    """
    Murrey Math Lines (VG / MT4) port.
    df: columns: time, open, high, low, close (time может быть, но не обязателен)
    Возвращает dict[str, pd.Series] с ключами mur_-2_8 ... mur_10_8 + zone/pos/dist
    """

    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close = df["close"].astype(float)

    print("Murrey DEBUF: compute_murrey_grid_mt4_vg")

    P = int(period_bars)

    # MT4: iLowest/iHighest(P, shift) -> окно "текущий бар и P-1 прошлых"
    v1 = low.rolling(P, min_periods=P).min()
    v2 = high.rolling(P, min_periods=P).max()

    rng = (v2 - v1)
    rng = rng.where(rng > 0)

    # --- fractal selection exactly like MQL4 (based on v2 absolute value) ---
    v2v = v2.values
    fractal = np.zeros_like(v2v, dtype=float)

    # translate chain of ifs literally
    fractal[(v2v <= 250000) & (v2v > 25000)] = 100000
    fractal[(v2v <= 25000)  & (v2v > 2500)]  = 10000
    fractal[(v2v <= 2500)   & (v2v > 250)]   = 1000
    fractal[(v2v <= 250)    & (v2v > 25)]    = 100
    fractal[(v2v <= 25)     & (v2v > 12.5)]  = 12.5
    fractal[(v2v <= 12.5)   & (v2v > 6.25)]  = 12.5
    fractal[(v2v <= 6.25)   & (v2v > 3.125)] = 6.25
    fractal[(v2v <= 3.125)  & (v2v > 1.5625)] = 3.125
    fractal[(v2v <= 1.5625) & (v2v > 0.390625)] = 1.5625
    fractal[(v2v <= 0.390625) & (v2v > 0)] = 0.1953125

    fractal_s = pd.Series(fractal, index=df.index)
    fractal_s = fractal_s.where(fractal_s > 0)

    # sum = floor(log(fractal/range)/log(2))
    ratio = (fractal_s / rng)
    ratio = ratio.where(ratio > 0)

    sum_pow = np.floor(np.log(ratio) / np.log(2.0))
    octave = fractal_s * (np.power(0.5, sum_pow))

    # mn = floor(v1/octave)*octave
    mn = np.floor(v1 / octave) * octave

    # mx = (mn+octave > v2) ? mn+octave : mn+2*octave
    mx = pd.Series(np.where((mn + octave) > v2, mn + octave, mn + 2.0 * octave), index=df.index)

    # helpers
    diff = (mx - mn)

    # --- calculating x's (exact conditions) ---
    x2 = pd.Series(np.where((v1 >= (3*diff/16 + mn)) & (v2 <= (9*diff/16 + mn)),
                            mn + diff/2, 0.0), index=df.index)

    x1 = pd.Series(np.where((v1 >= (mn - diff/8)) & (v2 <= (5*diff/8 + mn)) & (x2 == 0),
                            mn + diff/2, 0.0), index=df.index)

    x4 = pd.Series(np.where((v1 >= (mn + 7*diff/16)) & (v2 <= (13*diff/16 + mn)),
                            mn + 3*diff/4, 0.0), index=df.index)

    x5 = pd.Series(np.where((v1 >= (mn + 3*diff/8)) & (v2 <= (9*diff/8 + mn)) & (x4 == 0),
                            mx, 0.0), index=df.index)

    x3 = pd.Series(np.where((v1 >= (mn + diff/8)) & (v2 <= (7*diff/8 + mn)) &
                            (x1 == 0) & (x2 == 0) & (x4 == 0) & (x5 == 0),
                            mn + 3*diff/4, 0.0), index=df.index)

    x6 = pd.Series(np.where((x1 + x2 + x3 + x4 + x5) == 0, mx, 0.0), index=df.index)

    finalH = x1 + x2 + x3 + x4 + x5 + x6

    # --- calculating y's (exact conditions) ---
    y1 = pd.Series(np.where(x1 > 0, mn, 0.0), index=df.index)
    y2 = pd.Series(np.where(x2 > 0, mn + diff/4, 0.0), index=df.index)
    y3 = pd.Series(np.where(x3 > 0, mn + diff/4, 0.0), index=df.index)
    y4 = pd.Series(np.where(x4 > 0, mn + diff/2, 0.0), index=df.index)
    y5 = pd.Series(np.where(x5 > 0, mn + diff/2, 0.0), index=df.index)

    y6 = pd.Series(np.where((finalH > 0) & ((y1 + y2 + y3 + y4 + y5) == 0), mn, 0.0), index=df.index)

    finalL = y1 + y2 + y3 + y4 + y5 + y6

    print(
        f"[Murrey DEBUG] v1={v1.iloc[-1]:.6f} v2={v2.iloc[-1]:.6f} fractal={fractal_s.iloc[-1]:.2f} octave={octave.iloc[-1]:.10f} mn={mn.iloc[-1]:.6f} mx={mx.iloc[-1]:.6f} finalL={finalL.iloc[-1]:.6f} finalH={finalH.iloc[-1]:.6f}")

    dmml = (finalH - finalL) / 8.0

    # mml[0] = finalL - dmml*2  (это -2/8)
    mml0 = finalL - dmml*2.0

    # levels -2..+2 => total 13 lines (as in MT4)
    levels = {}
    for i in range(13):
        levels_i = mml0 + dmml * i
        # i=0 -> -2/8, i=2 -> 0/8, i=10 -> 8/8, i=12 -> +2/8
        mur_idx = i - 2
        levels[f"mur_{mur_idx}_8"] = levels_i

    # zone / position within zone based on 0/8..8/8 (same idea as your текущие фичи)
    l0 = levels["mur_0_8"]
    l8 = levels["mur_8_8"]

    # zone is between k/8 and (k+1)/8, k in [0..7]
    zone = pd.Series(np.nan, index=df.index, dtype=float)
    pos  = pd.Series(np.nan, index=df.index, dtype=float)

    valid = l0.notna() & l8.notna()
    if valid.any():
        # stack 0..8
        grid = np.vstack([levels[f"mur_{k}_8"].values for k in range(0, 9)])  # shape (9, n)
        c = close.values

        # find k such that grid[k] <= c < grid[k+1]
        # handle outside by clipping
        # compute k via count of levels <= close
        le_cnt = np.sum(grid <= c, axis=0) - 1  # gives -1..8
        k = np.clip(le_cnt, 0, 7)

        zone.loc[valid] = k[valid.values]

        # pos within zone
        lo = grid[k, np.arange(len(c))]
        hi = grid[k+1, np.arange(len(c))]
        denom = (hi - lo)
        denom = np.where(denom == 0, np.nan, denom)
        p = (c - lo) / denom
        pos.loc[valid] = p[valid.values]

    # distances (как у тебя)
    # nearest among 0..8
    nearest_idx = pd.Series(np.nan, index=df.index, dtype=float)
    dist_nearest = pd.Series(np.nan, index=df.index, dtype=float)

    if valid.any():
        grid0_8 = np.vstack([levels[f"mur_{k}_8"].values for k in range(0, 9)])  # (9,n)
        c = close.values
        d = np.abs(grid0_8 - c)
        ni = np.argmin(d, axis=0)
        nearest_idx.loc[valid] = ni[valid.values]
        dist_nearest.loc[valid] = d[ni, np.arange(len(c))][valid.values]

    out = dict(levels)

    out["mur_zone"] = zone
    out["mur_pos_in_zone"] = pos
    out["mur_nearest_idx"] = nearest_idx
    out["mur_dist_close_to_nearest"] = dist_nearest
    out["mur_dist_close_to_0_8"] = (close - levels["mur_0_8"]).abs()
    out["mur_dist_close_to_4_8"] = (close - levels["mur_4_8"]).abs()
    out["mur_dist_close_to_8_8"] = (close - levels["mur_8_8"]).abs()

    if not include_extremes:
        # если экстремы не нужны — просто удалим -2,-1,9,10
        for k in (-2, -1, 9, 10):
            out.pop(f"mur_{k}_8", None)

    return out

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

    print("Murrey DEBUF: compute_murrey_grid_mt4_exact")

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
