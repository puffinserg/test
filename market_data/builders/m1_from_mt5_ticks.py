# market_data/builders/m1_from_mt5_ticks.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import SETTINGS
from config.paths import EXTERNAL_DIR
from market_data.connectors.mt5_connector import init_mt5, shutdown_mt5, fetch_ticks_range
from numpy.ma.extras import apply_along_axis
from sympy.physics.units import kelvin


def _parse_utc(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        return datetime(2010, 1, 1, tzinfo=timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _month_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m")


def _year_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y")


def _m1_file_path(symbol: str, ts: pd.Timestamp, granularity: str) -> Path:
    if granularity == "year":
        return EXTERNAL_DIR / f"{symbol}_M1_{_year_key(ts)}.parquet"
    return EXTERNAL_DIR / f"{symbol}_M1_{_month_key(ts)}.parquet"

def _mt5_server_offset_hours() -> int:
    """
    Возвращает смещение серверного времени MT5 относительно UTC в часах.
    Работает и при DST: вычисляется "сейчас" через time_current().
    """
    server_ts = mt5.time_current()
    if not server_ts:
        raise RuntimeError("mt5.time_current() returned 0/None (not connected?)")

    # server_ts — это 'server time' в виде Unix timestamp.
    # Сравниваем с реальным UTC now, получаем offset.
    server_dt = datetime.fromtimestamp(server_ts, tz=timezone.utc)
    utc_now = datetime.now(timezone.utc)

    offset_sec = (server_dt - utc_now).total_seconds()
    return int(round(offset_sec / 3600))

def ticks_to_m1_bid(df_ticks: pd.DataFrame) -> pd.DataFrame:
    """
    M1 по BID:
      open/high/low/close = bid
      spread_mean/spread_max = ask-bid
    """
    if df_ticks is None or df_ticks.empty:
        return pd.DataFrame()

    df = df_ticks.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").set_index("time")

    if "bid" not in df.columns:
        return pd.DataFrame()

    if "ask" in df.columns:
        df["spread"] = (df["ask"] - df["bid"]).astype("float64")
    else:
        df["spread"] = 0.0

    ohlc = df["bid"].resample("1min", label="left", closed="left").ohlc()
    spread_mean = df["spread"].resample("1min", label="left", closed="left").mean()
    spread_max = df["spread"].resample("1min", label="left", closed="left").max()

    out = ohlc.join(spread_mean.rename("spread_mean")).join(spread_max.rename("spread_max"))
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def merge_write_m1(symbol: str, m1: pd.DataFrame, granularity: str) -> None:
    if m1.empty:
        return

    m1["time"] = pd.to_datetime(m1["time"], utc=True)

    # группируем по месяцам/годам и пишем по файлам
    grp = m1["time"].dt.strftime("%Y%m") if granularity == "month" else m1["time"].dt.strftime("%Y")
    for key, chunk in m1.groupby(grp):
        ts0 = pd.to_datetime(chunk["time"].iloc[0], utc=True)
        out_path = _m1_file_path(symbol, ts0, granularity)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            old = pd.read_parquet(out_path)
            if not old.empty:
                old["time"] = pd.to_datetime(old["time"], utc=True)
                merged = (
                    pd.concat([old, chunk], ignore_index=True)
                    .drop_duplicates(subset=["time"], keep="last")
                    .sort_values("time")
                    .reset_index(drop=True)
                )
            else:
                merged = chunk.sort_values("time").reset_index(drop=True)
        else:
            merged = chunk.sort_values("time").reset_index(drop=True)

        merged.to_parquet(out_path, index=False)
        print(f"[mt5->m1] wrote {len(merged)} rows -> {out_path.name}")


def build_external_m1_from_mt5_ticks_backfill() -> None:
    """
    Качаем тики из MT5 чанками и сразу строим external M1 в EXTERNAL_DIR (активный source=mt5).
    Повторный запуск безопасен: merge по time.
    """
    cfg = getattr(SETTINGS, "mt5", None)
    if cfg is None:
        print("[mt5->m1] No mt5 section in settings.yaml")
        return

    symbol = cfg.symbol or SETTINGS.market.symbol
    start_utc = _parse_utc(getattr(cfg, "ticks_start_utc", "2020-06-01T00:00:00Z"))
    chunk_hours = int(getattr(cfg, "chunk_hours", 24))
    empty_stop = int(getattr(cfg, "empty_chunks_stop", 50))
    gran = str(getattr(cfg, "m1_granularity", "month")).strip().lower()

    # end = "сейчас": якорим на 00:00 серверного дня (в UTC)
    init_mt5()  # важно: сначала подключиться к MT5
    offset_h = _mt5_server_offset_hours()
    utc_now = datetime.now(timezone.utc)
    server_now = utc_now + timedelta(hours=offset_h)
    server_midnight = server_now.replace(hour=0, minute=0, second=0, microsecond=0)
    dt_to = server_midnight - timedelta(hours=offset_h)
    print(f"[mt5->m1] detected server_offset_hours={offset_h}, dt_to_utc={dt_to.isoformat()}")

    delta = timedelta(hours=chunk_hours)

    print(f"[mt5->m1] Backfill {symbol}: {start_utc.isoformat()} -> {dt_to.isoformat()}")
    print(f"[mt5->m1] chunk_hours={chunk_hours}, empty_chunks_stop={empty_stop}, granularity={gran}")
    print(f"[mt5->m1] external_dir={EXTERNAL_DIR}")

    empty_streak = 0
    chunk_idx = 0

    # init_mt5() уже вызван выше перед расчетом dt_to
    try:

        while dt_to > start_utc:
            # На длинном backfill пересчитываем offset иногда (DST/смены сервера)
            if chunk_idx % 14 == 0:  # раз в 14 чанков (примерно раз в 2 недели при chunk_hours=24)
                offset_h = _mt5_server_offset_hours()
            chunk_idx += 1
            dt_from = dt_to - delta
            if dt_from < start_utc:
                dt_from = start_utc

            df_ticks = fetch_ticks_range(symbol, dt_from, dt_to)

            if df_ticks.empty:
                empty_streak += 1
                print(f"[mt5->m1] {dt_from} -> {dt_to}: 0 ticks (empty_streak={empty_streak})")
                if empty_streak >= empty_stop:
                    print("[mt5->m1] Stop: too many empty chunks подряд. История, вероятно, закончилась.")
                    break
            else:
                empty_streak = 0
                print(f"[mt5->m1] {dt_from} -> {dt_to}: {len(df_ticks)} ticks")
                m1 = ticks_to_m1_bid(df_ticks)
                merge_write_m1(symbol, m1, gran)

            dt_to = dt_from

    finally:
        shutdown_mt5()

    print("[mt5->m1] Done.")
