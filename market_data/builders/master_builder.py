# RL_PyTorch/market_data/builders/master_builder.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.paths import MASTER_DIR, EXTERNAL_DIR, LIVE_LOG_DIR
from config.settings import SETTINGS
from market_data.connectors.mt5_connector import (
    init_mt5,
    shutdown_mt5,
    download_m1_history,
)

def _master_file_path() -> Path:
    symbol = SETTINGS.market.symbol
    tf = SETTINGS.market.timeframe
    pattern = SETTINGS.paths.master_pattern
    filename = pattern.format(symbol=symbol, tf=tf)
    return MASTER_DIR / filename

def _print_history_range(path: Path) -> None:
    """
    Читает только колонку time и печатает период истории.
    """
    try:
        df = pd.read_parquet(path, columns=["time"])
    except Exception as e:
        print(f"[master_builder] Не удалось прочитать {path}: {e}")
        return

    if df.empty:
        print(f"[master_builder] Файл {path} пустой")
        return

    t_start = df["time"].iloc[0]
    t_end = df["time"].iloc[-1]
    print(f"[master_builder] История в master-файле: {t_start} -> {t_end} "
          f"(строк: {len(df)})")

def build_master_from_external_impl() -> None:
    """
    Собирает master-историю из external_dir активного источника (data_source).
    Ожидает файлы формата:
        <symbol>_<TF>_YYYY.parquet
    Например:
        EURUSD_M1_2010.parquet
    """

    symbol = SETTINGS.market.symbol
    tf = SETTINGS.market.timeframe

    external_dir = EXTERNAL_DIR
    pattern = f"{symbol}_{tf}_*.parquet"

    if not external_dir.exists():
        print(f"[master_builder] External dir not found: {external_dir}")
        return

    files = sorted(external_dir.glob(pattern))
    if not files:
        print(f"[master_builder] Не найдено файлов по шаблону {pattern} в {external_dir}")
        return

    print(f"[master_builder] Собираем master из external (source={getattr(SETTINGS,'data_source','?')}) для {symbol}, TF={tf}")
    print("  Найдены файлы:")
    for f in files:
        print("   ", f.name)

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"[master_builder] ERROR чтения {f}: {e}")
            continue
        if df.empty:
            print(f"[master_builder] WARNING: файл пустой {f}")
            continue
        dfs.append(df)

    if not dfs:
        print("[master_builder] Нет валидных данных для сборки master.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # Приводим к единому виду: сортируем, убираем дубликаты по времени
    if "time" not in df_all.columns:
        print("[master_builder] ERROR: нет колонки 'time' в данных!")
        return

    df_all = (
        df_all
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    path = _master_file_path()
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[master_builder] Сохраняем master ({len(df_all)} строк) -> {path}")
    df_all.to_parquet(path, index=False)

    _print_history_range(path)

def update_master_from_live_logs_impl() -> None:
    path = _master_file_path()
    if not path.exists():
        print(f"[master_builder] Master file not found: {path}")
        return

    # 1. Загружаем существующий master
    df_master = pd.read_parquet(path)
    if df_master.empty or "time" not in df_master.columns:
        print("[master_builder] Master пустой или без 'time' – обновлять нечего.")
        return

    df_master["time"] = pd.to_datetime(df_master["time"], utc=True)
    df_master = df_master.sort_values("time").reset_index(drop=True)
    last_time = df_master["time"].iloc[-1]
    print(f"[master_builder] Последний бар master: {last_time}")

    # 2. Собираем все live-логи
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_files = sorted(LIVE_LOG_DIR.glob("*.parquet"))
    if not log_files:
        print(f"[master_builder] Нет файлов live-логов в {LIVE_LOG_DIR}")
        return

    dfs_new = []
    for f in log_files:
        try:
            df_log = pd.read_parquet(f)
        except Exception as e:
            print(f"[master_builder] ERROR чтения {f}: {e}")
            continue

        if df_log.empty or "time" not in df_log.columns:
            print(f"[master_builder] WARNING: {f} пустой или без 'time'")
            continue

        df_log["time"] = pd.to_datetime(df_log["time"], utc=True)
        df_log = df_log[df_log["time"] > last_time]
        if not df_log.empty:
            dfs_new.append(df_log)

    if not dfs_new:
        print("[master_builder] Нет баров новее текущего master.")
        return

    df_new = (
        pd.concat(dfs_new, ignore_index=True)
          .drop_duplicates(subset=["time"])
          .sort_values("time")
          .reset_index(drop=True)
    )
    print(f"[master_builder] Новых баров: {len(df_new)}")

    # 3. Объединяем и сохраняем master
    df_all = (
        pd.concat([df_master, df_new], ignore_index=True)
          .drop_duplicates(subset=["time"])
          .sort_values("time")
          .reset_index(drop=True)
    )

    print(f"[master_builder] Новый размер master: {len(df_all)} строк. Сохраняем -> {path}")
    df_all.to_parquet(path, index=False)
    _print_history_range(path)

