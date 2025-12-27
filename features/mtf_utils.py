# features/mtf_utils.py

from __future__ import annotations
from typing import List, Optional
import pandas as pd

TF_TO_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

TF_TO_PANDAS = {
    "M1": "1T",
    "M5": "5T",
    "M15": "15T",
    "M30": "30T",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def tf_factor(tf: str, base_tf: str) -> float:
    base = TF_TO_MINUTES.get(base_tf)
    cur = TF_TO_MINUTES.get(tf)
    if base is None or cur is None or base == 0:
        return 1.0
    return cur / base


def resample_ohlc(
    df: pd.DataFrame,
    tf: str,
    time_col: str = "time",
    ohlc_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ресемплинг H1 -> H4/D1 и т.п.
    Делает классический OHLC-агрегат по указанному таймфрейму.
    """
    if ohlc_cols is None:
        ohlc_cols = ["open", "high", "low", "close"]

    rule = TF_TO_PANDAS.get(tf)
    if rule is None:
        raise ValueError(f"Unsupported TF for resample: {tf}")

    df_idx = df.set_index(time_col)

    agg_dict = {}
    if "open" in ohlc_cols:
        agg_dict["open"] = "first"
    if "high" in ohlc_cols:
        agg_dict["high"] = "max"
    if "low" in ohlc_cols:
        agg_dict["low"] = "min"
    if "close" in ohlc_cols:
        agg_dict["close"] = "last"

    res = (
        # Совместимо с тем, как формируются H1 снапшоты в snapshot_builder
        # и с тем, как MT4/MT5 маркируют бар временем его открытия.
        df_idx.resample(rule, label="left", closed="left")
        .agg(agg_dict)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
        .rename(columns={time_col: "time"})
    )
    return res


def align_higher_tf_to_working(
    df_work: pd.DataFrame,
    df_htf: pd.DataFrame,
    cols: List[str],
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Подтянуть значения со старшего TF к рабочему через merge_asof.
    Для каждого бара рабочего TF берется последний полностью сформированный бар старшего TF.
    """
    def _ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
        # Если time уже колонка — ок
        if time_col in df.columns:
            return df
        # Если time в DatetimeIndex — вынесем в колонку
        if isinstance(df.index, pd.DatetimeIndex):
            out = df.copy()
            idx_name = out.index.name or time_col
            out = out.reset_index().rename(columns={idx_name: time_col})
            return out
        raise KeyError(f"align_higher_tf_to_working: '{time_col}' not found in columns and index is not DatetimeIndex")

    dfw = _ensure_time_column(df_work).sort_values(time_col)
    dfh = _ensure_time_column(df_htf).sort_values(time_col)[[time_col] + cols]

    # --- normalize time dtype (critical for merge_asof reliability) ---
    dfw = dfw.copy()
    dfh = dfh.copy()
    dfw[time_col] = pd.to_datetime(dfw[time_col], errors="coerce")
    dfh[time_col] = pd.to_datetime(dfh[time_col], errors="coerce")
    # remove timezone if present
    if getattr(dfw[time_col].dt, "tz", None) is not None:
        dfw[time_col] = dfw[time_col].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        dfw[time_col] = dfw[time_col].dt.tz_localize(None)
    if getattr(dfh[time_col].dt, "tz", None) is not None:
        dfh[time_col] = dfh[time_col].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        dfh[time_col] = dfh[time_col].dt.tz_localize(None)

    dfw = dfw.sort_values(time_col)
    dfh = dfh.sort_values(time_col)

    merged = pd.merge_asof(
        dfw,
        dfh,
        on=time_col,
        direction="backward",
    )
    return merged