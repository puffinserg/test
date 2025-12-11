from __future__ import annotations

import datetime as dt
import struct
import lzma
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from config.settings import SETTINGS

import pandas as pd
import requests


DUKASCOPY_BASE_URL = "https://datafeed.dukascopy.com/datafeed"
# Формат свечи: Seconds, O, H, L, C, V
CANDLE_STRUCT_FMT = ">IIIIIf"
CANDLE_SIZE = struct.calcsize(CANDLE_STRUCT_FMT)


@dataclass
class DukascopyConfig:
    symbol: str = "EURUSD"
    price_divisor: int = 100_000   # для форекса (1.14271 = 114271 / 1e5)
    timeout_sec: int = 10
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )


def _build_m1_url(symbol: str, date: dt.date) -> str:
    """
    Формирует URL вида:
    https://datafeed.dukascopy.com/datafeed/EURUSD/2019/05/01/BID_candles_min_1.bi5

    ВАЖНО: месяц в URL 0-based, поэтому month - 1.
    """
    year = date.year
    month0 = date.month - 1
    day = date.day
    return (
        f"{DUKASCOPY_BASE_URL}/{symbol.upper()}/"
        f"{year:04d}/{month0:02d}/{day:02d}/BID_candles_min_1.bi5"
    )


def _download_bi5(url: str, cfg: DukascopyConfig) -> Optional[bytes]:
    headers = {"User-Agent": cfg.user_agent}
    resp = requests.get(url, headers=headers, timeout=cfg.timeout_sec)

    if resp.status_code == 404 or len(resp.content) == 0:
        # Нет данных на эту дату
        return None

    resp.raise_for_status()
    return resp.content


def _parse_candles_bi5(raw: bytes, base_date: dt.date, cfg: DukascopyConfig) -> pd.DataFrame:
    """
    raw – lzma-сжатый bi5, формат >IIIIIf:
    Seconds, O, H, L, C, V
    """
    decompressed = lzma.decompress(raw)
    n_records = len(decompressed) // CANDLE_SIZE

    records: List[tuple] = []
    for i in range(n_records):
        chunk = decompressed[i * CANDLE_SIZE : (i + 1) * CANDLE_SIZE]
        sec, o, h, l, c, v = struct.unpack(CANDLE_STRUCT_FMT, chunk)

        # timestamp = начало дня (UTC) + sec
        ts = dt.datetime(base_date.year, base_date.month, base_date.day, tzinfo=dt.timezone.utc) \
             + dt.timedelta(seconds=sec)

        price_div = cfg.price_divisor
        records.append(
            (
                ts,
                o / price_div,
                h / price_div,
                l / price_div,
                c / price_div,
                float(v),
            )
        )

    df = pd.DataFrame(
        records,
        columns=["time", "open", "high", "low", "close", "tick_volume"],
    )

    # Для совместимости с MT5-данными:
    df["spread"] = 0
    df["real_volume"] = 0

    return df

def download_history_from_settings() -> None:
    """
    Качает историю M1 с Dukascopy по параметрам из SETTINGS.dukascopy.
    Кладёт годовые parquet в external_dir.
    """
    dcfg = SETTINGS.dukascopy
    symbol = dcfg.symbol or SETTINGS.market.symbol

    base_dir = Path(__file__).resolve().parents[2]  # .../RL_PyTorch/
    external_dir = base_dir / dcfg.external_dir

    cfg = DukascopyConfig(
        symbol=symbol,
        price_divisor=dcfg.price_divisor,
    )

    start_year = dcfg.start_year
    years = dcfg.years

    today = dt.date.today()
    max_year = start_year + years  # не выходим за пределы
    current_year = min(today.year, max_year)

    # 1) Полные года
    for year in range(start_year, current_year):
        start = dt.date(year, 1, 1)
        end = dt.date(year + 1, 1, 1)

        out_file = external_dir / f"{symbol}_M1_{year}.parquet"
        if out_file.exists():
            print(f"[dukascopy-main] Год {year}: файл уже есть, пропускаем -> {out_file}")
            continue

        print(f"\n[dukascopy-main] ===== Год {year}: {start} .. {end} -> {out_file}")
        download_m1_range_to_parquet(start, end, out_file, cfg)

    # 2) Текущий (частичный) год
    start = dt.date(current_year, 1, 1)
    end = min(
        today + dt.timedelta(days=1),
        dt.date(start_year + years, 1, 1)
    )

    out_file = external_dir / f"{symbol}_M1_{current_year}.parquet"
    if out_file.exists():
        print(f"[dukascopy-main] Текущий год {current_year}: файл уже есть, пропускаем -> {out_file}")
    else:
        print(f"\n[dukascopy-main] ===== Текущий год {current_year}: {start} .. {end} -> {out_file}")
        download_m1_range_to_parquet(start, end, out_file, cfg)

def download_m1_range_to_df(
    start_date: dt.date,
    end_date: dt.date,
    cfg: DukascopyConfig,
) -> pd.DataFrame:
    """
    Скачивает M1-данные за [start_date, end_date) включительно-исключительно
    и возвращает единый DataFrame.
    """
    all_chunks: List[pd.DataFrame] = []
    current = start_date

    while current < end_date:
        url = _build_m1_url(cfg.symbol, current)
        print(f"[dukascopy] {current} -> {url}")
        try:
            raw = _download_bi5(url, cfg)
        except Exception as e:
            print(f"[dukascopy] ERROR {current}: {e}")
            current += dt.timedelta(days=1)
            continue

        if raw is None:
            print(f"[dukascopy] no data for {current}")
        else:
            df_day = _parse_candles_bi5(raw, current, cfg)
            all_chunks.append(df_day)
            print(f"[dukascopy] {current}: {len(df_day)} rows")

        current += dt.timedelta(days=1)

    if not all_chunks:
        raise RuntimeError("Не удалось получить ни одного дня данных")

    df_all = pd.concat(all_chunks, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df_all

def download_m1_range_to_parquet(
    start_date: dt.date,
    end_date: dt.date,
    out_path: Path,
    cfg: DukascopyConfig,
) -> None:
    df = download_m1_range_to_df(start_date, end_date, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[dukascopy] Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    download_history_from_settings()



