# RL_PyTorch/market_data/builders/snapshot_builder.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from config.settings import SETTINGS
from config.paths import SNAPSHOT_DIR
from .master_builder import _master_file_path


# ---------- вспомогательные функции ----------
def _current_snapshot_tf() -> str:
    """
    Рабочий ТФ для снапшотов.
    Если есть SETTINGS.market.working_timeframe — используем его,
    иначе по умолчанию берём master TF.
    """
    return getattr(SETTINGS.market, "working_timeframe", SETTINGS.market.timeframe)

def _snapshot_path(tf: str, role: str, start_date: str, end_date: str) -> Path:
    """
    Имя файла снапшота по paths.snapshot_pattern.
    Пример pattern:
      {symbol}_{tf}_{role}_snapshot_{start}_{end}.parquet
    """
    symbol = SETTINGS.market.symbol
    pattern = SETTINGS.paths.snapshot_pattern
    filename = pattern.format(
        symbol=symbol,
        tf=tf,
        role=role,
        start=start_date,
        end=end_date,
    )
    return SNAPSHOT_DIR / filename



def _list_snapshot_files() -> List[Path]:
    """
    Возвращает список snapshot-файлов для текущего символа.
    TF может быть разным, поэтому фильтруем только по symbol и 'snapshot'.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    symbol = SETTINGS.market.symbol
    tf = _current_snapshot_tf()
    pattern = f"{symbol}_{tf}_*snapshot_*.parquet"
    files = sorted(SNAPSHOT_DIR.glob(pattern))

    return list(files)


def _parse_snapshot_name(path: Path) -> Tuple[str, str, str, str]:
    """
    Пытается вытащить (tf, role, start_date, end_date) из имени файла.

    Новый формат:
        <symbol>_<tf>_<role>_snapshot_<start>_<end>.parquet

    Пример:
        EURUSD_H1_train_snapshot_2010-01-01_2018-12-31.parquet
        [0]   [1] [2]   [3]       [4]         [5]
    """
    stem = path.stem
    parts = stem.split("_")

    tf = "?"
    role = "unknown"
    start_date = "?"
    end_date = "?"

    # минимальная длина: symbol, tf, role, snapshot, start, end  => 6 частей
    if len(parts) < 6:
        return tf, role, start_date, end_date

    # parts[0] = symbol
    tf = parts[1]
    role = parts[2]

    # ищем 'snapshot'
    try:
        snap_idx = parts.index("snapshot")
    except ValueError:
        return tf, role, start_date, end_date

    if snap_idx + 2 < len(parts):
        start_date = parts[snap_idx + 1]
        end_date = parts[snap_idx + 2]

    return tf, role, start_date, end_date


def _tf_to_pandas_freq(tf: str) -> str:
    """
    Маппинг строкового TF на частоту pandas.
    """
    mapping = {
        "M1": "1T",
        "M5": "5T",
        "M15": "15T",
        "M30": "30T",
        "H1": "1H",
        "H4": "4H",
        "D1": "1D",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[tf].lower()

def _resample_m1_to_tf(df_m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Агрегирует M1-данные до заданного TF.

    Ожидается, что df_m1 содержит:
        time, open, high, low, close, tick_volume, spread, real_volume

    Правила:
        open  = first
        high  = max
        low   = min
        close = last
        tick_volume = sum
        spread = mean
        real_volume = sum
    """
    if tf == "M1":
        # Без агрегации, просто возвращаем копию
        return df_m1.copy()

    freq = _tf_to_pandas_freq(tf)

    if "time" not in df_m1.columns:
        raise ValueError("df_m1 must contain 'time' column")

    df = df_m1.set_index("time")

    agg_dict = {}
    # Если есть отдельные колонки OHLC, агрегируем каждую
    if {"open", "high", "low", "close"}.issubset(df.columns):
        agg_dict.update({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
    else:
        # fallback: считаем, что есть только close
        agg_dict["close"] = "last"

    if "tick_volume" in df.columns:
        agg_dict["tick_volume"] = "sum"
    if "spread" in df.columns:
        agg_dict["spread"] = "mean"
    if "real_volume" in df.columns:
        agg_dict["real_volume"] = "sum"

    res = df.resample(freq).agg(agg_dict)
    res = res.dropna(how="any").reset_index()  # time вернётся в колонки

    return res

# ---------- создание snapshot’а ----------

def create_snapshot_from_master_impl(
    start_date: str,
    end_date: str,
    role: str,
    tf: str | None = None,
) -> None:
    """
    Создаёт snapshot по заданному диапазону дат и TF.

    Логика:
      1. Читаем master (M1).
      2. Подрезаем по пересечению с запрошенным диапазоном.
      3. Агрегируем M1 -> TF (M1/M5/M15/H1/...).
      4. Сохраняем "чистый" snapshot без фичей.
    """
    tf = tf or _current_snapshot_tf()
    master_path = _master_file_path()
    if not master_path.exists():
        print(f"[snapshot_builder] Master file not found: {master_path}")
        return

    # --- 1. Проверяем формат дат ---
    start_dt_naive = pd.to_datetime(start_date, format="%Y-%m-%d", errors="coerce")
    end_dt_naive = pd.to_datetime(end_date,   format="%Y-%m-%d", errors="coerce")

    if pd.isna(start_dt_naive) or pd.isna(end_dt_naive):
        print("[snapshot_builder] ERROR: неверный формат даты. "
              "Ожидается YYYY-MM-DD (например, 2012-01-01).")
        return

    if start_dt_naive > end_dt_naive:
        print("[snapshot_builder] ERROR: дата начала больше даты конца.")
        return

    # Переводим в UTC-aware
    start_dt_req = start_dt_naive.tz_localize("UTC")
    # конец дня: 23:59:59 для end_date
    end_dt_req = (end_dt_naive + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize("UTC")

    # --- 2. Загружаем master ---
    print(f"[snapshot_builder] Loading master (M1) from {master_path}")
    df = pd.read_parquet(master_path)

    if "time" not in df.columns:
        print("[snapshot_builder] ERROR: 'time' column not found in master!")
        return

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    master_start = df["time"].iloc[0]
    master_end   = df["time"].iloc[-1]
    print(f"[snapshot_builder] Master range: {master_start} -> {master_end}")

    # --- 3. Проверяем пересечение диапазонов ---
    if end_dt_req < master_start or start_dt_req > master_end:
        print("[snapshot_builder] ERROR: запрошенный диапазон полностью вне master-истории.")
        return

    eff_start = max(start_dt_req, master_start)
    eff_end   = min(end_dt_req,   master_end)

    if eff_start != start_dt_req or eff_end != end_dt_req:
        print("[snapshot_builder] WARNING: диапазон скорректирован до пересечения с master.")
        print(f"  Запрошено: {start_dt_req} -> {end_dt_req}")
        print(f"  Фактически: {eff_start} -> {eff_end}")

    eff_start_str = eff_start.strftime("%Y-%m-%d")
    eff_end_str   = eff_end.strftime("%Y-%m-%d")

    print(f"[snapshot_builder] Creating snapshot between {eff_start} and {eff_end}, "
          f"role={role}, tf={tf}")

    # --- 4. Режем master по времени ---
    mask = (df["time"] >= eff_start) & (df["time"] <= eff_end)
    df_slice_m1 = df.loc[mask].copy()

    if df_slice_m1.empty:
        print("[snapshot_builder] WARNING: no data in selected range.")
        return

    # --- 5. Агрегация M1 -> TF ---
    try:
        df_tf = _resample_m1_to_tf(df_slice_m1, tf)
    except ValueError as e:
        print(f"[snapshot_builder] ERROR during resample: {e}")
        return

    if df_tf.empty:
        print("[snapshot_builder] WARNING: aggregated dataframe is empty.")
        return

    # --- 6. Сохраняем snapshot ---
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap_path = _snapshot_path(tf, role, eff_start_str, eff_end_str)
    df_tf.to_parquet(snap_path, index=False)

    print(f"[snapshot_builder] Saved snapshot {len(df_tf)} rows -> {snap_path}")

# ---------- список snapshot’ов ----------

def list_snapshots_impl() -> List[Path]:
    """
    Печатает список snapshot’ов и возвращает их список.
    """
    files = _list_snapshot_files()
    if not files:
        print("[snapshot_builder] Snapshot files not found.")
        return []

    print("[snapshot_builder] Snapshot files:")
    for idx, path in enumerate(files, start=1):
        tf, role, start_date, end_date = _parse_snapshot_name(path)
        print(f"  {idx:2d}. TF={tf:<4} role={role:<6} {start_date} .. {end_date}   {path.name}")

    return files


# ---------- проверка snapshot’а ----------

def validate_snapshot_impl() -> None:
    """
    Позволяет выбрать snapshot и проверяет его целостность:
    - наличие колонки time
    - сортировку по времени
    - отсутствие дубликатов времён
    - равномерный шаг по времени
    - отсутствие NaN в основных колонках
    """
    files = list_snapshots_impl()
    if not files:
        return

    choice = input("Введите номер snapshot для проверки: ").strip()
    if not choice.isdigit():
        print("[snapshot_validator] Неверный ввод.")
        return

    idx = int(choice) - 1
    if not (0 <= idx < len(files)):
        print("[snapshot_validator] Неверный номер.")
        return

    path = files[idx]
    print(f"[snapshot_validator] Checking {path}")

    df = pd.read_parquet(path)

    print(f"[snapshot_validator] Rows: {len(df)}")
    if "time" not in df.columns:
        print("[snapshot_validator] ERROR: 'time' column not found!")
        return

    df["time"] = pd.to_datetime(df["time"], utc=True)

    # сортировка по времени
    if not df["time"].is_monotonic_increasing:
        print("[snapshot_validator] WARNING: 'time' is not sorted, sorting...")
        df = df.sort_values("time").reset_index(drop=True)

    # дубликаты
    dups = df["time"].duplicated()
    if dups.any():
        print(f"[snapshot_validator] WARNING: duplicated timestamps found: {dups.sum()}")
        print(df.loc[dups].head())
    else:
        print("[snapshot_validator] OK: no duplicated timestamps")

    # шаг по времени
    delta = df["time"].diff().dropna()
    unique_deltas = delta.unique()
    if len(unique_deltas) == 1:
        print(f"[snapshot_validator] OK: uniform TF = {unique_deltas[0]}")
    else:
        print(f"[snapshot_validator] WARNING: non-uniform time deltas → {unique_deltas}")

    # NaN в ключевых колонках (если они есть)
    required = ["open", "high", "low", "close"]
    existing = [c for c in required if c in df.columns]

    if existing:
        has_nan = df[existing].isna().any().any()
        if has_nan:
            print("[snapshot_validator] WARNING: NaNs detected in OHLC columns")
        else:
            print("[snapshot_validator] OK: no NaNs in OHLC columns")
    else:
        print("[snapshot_validator] WARNING: OHLC columns not fully present:", required)

    # === превью снапшота ===
    try:
        print("\n[snapshot_validator] Preview of snapshot (first 3 rows):")
        print(df.head(3))

        print("\n[snapshot_validator] Preview of snapshot (last 3 rows):")
        print(df.tail(3))
    except Exception as e:
        print(f"[snapshot_validator] ERROR printing preview: {e}")

# ---------- удаление snapshot’а ----------

def delete_snapshot_impl() -> None:
    """
    Позволяет выбрать и удалить snapshot-файл.
    """
    files = list_snapshots_impl()
    if not files:
        return

    choice = input("Введите номер snapshot для УДАЛЕНИЯ: ").strip()
    if not choice.isdigit():
        print("[snapshot_builder] Неверный ввод.")
        return

    idx = int(choice) - 1
    if not (0 <= idx < len(files)):
        print("[snapshot_builder] Неверный номер.")
        return

    path = files[idx]
    confirm = input(f"Точно удалить файл?\n  {path}\n(yes/no): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("[snapshot_builder] Отмена удаления.")
        return

    path.unlink()
    print(f"[snapshot_builder] Deleted snapshot file: {path}")
