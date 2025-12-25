# RL_PyTorch/market_data/providers/training_provider.py
"""
Обёртки для запуска training/validation из main.py.

Сейчас:
- интерактивный выбор snapshot из data/snapshots
- выбор профиля фич
- расчёт фич и вывод feature_dim и последнего вектора признаков.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

from config.settings import SETTINGS
from config.paths import SNAPSHOT_DIR, MASTER_DIR
from features.feature_engine import FeatureEngine
from datetime import timedelta

# ВАЖНО: используем тот же ресемплер, что и при создании snapshot (чтобы не было расхождений)
from market_data.builders.snapshot_builder import _resample_m1_to_tf  # noqa: WPS450 (private import, но минимальный патч)

# ---------- вспомогательные функции ----------
def _parse_snapshot_name(path: Path) -> Optional[Dict[str, str]]:
    """
    Ожидаемый формат имени:
      {symbol}_{tf}_{role}_snapshot_{start}_{end}.parquet
    Например:
      EURUSD_H1_train_snapshot_2012-01-01_2015-12-31.parquet
    """
    stem = path.name
    if stem.endswith(".parquet"):
        stem = stem[:-8]

    parts = stem.split("_")
    if len(parts) < 6:
        return None

    symbol = parts[0]
    tf = parts[1]
    role = parts[2]

    return {"symbol": symbol, "tf": tf, "role": role}

def _list_snapshots(role: Optional[str] = None,
                    tf: Optional[str] = None) -> List[Path]:
    """
    Возвращает список snapshot-файлов.

    Фильтруем по:
      - role (train/valid/test/other)
      - tf (H1/M1 и т.д.)
    """
    if not SNAPSHOT_DIR.exists():
        return []

    files = sorted(SNAPSHOT_DIR.glob("*.parquet"))
    result: List[Path] = []

    for p in files:
        meta = _parse_snapshot_name(p)
        if meta is None:
            continue
        if role and meta["role"] != role:
            continue
        if tf and meta["tf"] != tf:
            continue
        result.append(p)

    return result

def _choose_snapshot_interactive(role: Optional[str] = None,
                                 tf: Optional[str] = None) -> Optional[Path]:
    files = _list_snapshots(role=role, tf=tf)
    if not files:
        extra = []
        if role:
            extra.append(f"role={role}")
        if tf:
            extra.append(f"tf={tf}")
        extra_str = " (" + ", ".join(extra) + ")" if extra else ""
        print(f"[Training] В каталоге {SNAPSHOT_DIR} нет snapshot-файлов{extra_str}.")
        return None

    print(f"[Training] Доступные snapshot-файлы в {SNAPSHOT_DIR}:")
    for i, path in enumerate(files, start=1):
        meta = _parse_snapshot_name(path) or {}
        role_s = meta.get("role", "?")
        tf_s = meta.get("tf", "?")
        print(f" {i}. [{tf_s}/{role_s}] {path.name}")

    choice = input("Выберите номер snapshot (0 = отмена): ").strip()
    if not choice or not choice.isdigit():
        print("[Training] Отмена.")
        return None

    idx = int(choice)
    if idx == 0:
        print("[Training] Отмена.")
        return None
    if not (1 <= idx <= len(files)):
        print("[Training] Неверный номер.")
        return None

    return files[idx - 1]

def _select_feature_profile() -> str:
    cfg = SETTINGS.features
    profiles = list(cfg.profiles.keys())

    if not profiles:
        print(f"[Training] Профили фич не заданы, используем default_profile={cfg.default_profile}")
        return cfg.default_profile

    print("[Training] Доступные профили фич:")
    for i, name in enumerate(profiles, start=1):
        print(f"  {i}. {name}")

    choice = input(f"Выберите профиль (Enter – {cfg.default_profile}): ").strip()

    if not choice:
        return cfg.default_profile

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(profiles):
            return profiles[idx - 1]

    if choice in cfg.profiles:
        return choice

    print(f"[Training] Профиль '{choice}' не найден, используем {cfg.default_profile}")
    return cfg.default_profile

def _tf_to_minutes(tf: str) -> int:
    tf = tf.upper().strip()
    if tf == "M1": return 1
    if tf == "M5": return 5
    if tf == "M15": return 15
    if tf == "M30": return 30
    if tf == "H1": return 60
    if tf == "H4": return 240
    if tf == "D1": return 1440
    raise ValueError(f"Unsupported timeframe: {tf}")

def _collect_required_tfs(settings) -> list[str]:
    """
    Собираем TF, которые реально нужны для расчёта фич.
    - working_tf из market
    - tfs из MTF-фич (supertrend/murrey)
    """
    tfs = {SETTINGS.market.working_timeframe.upper()}
    for x in (SETTINGS.features.supertrend.tfs or []):
        tfs.add(str(x).upper())
    for x in (SETTINGS.features.murrey.tfs or []):
        tfs.add(str(x).upper())

    # на будущее: если появятся другие MTF-фичи — добавишь их сюда по аналогии
    return sorted(tfs)

# ---------- публичные функции ----------

def run_training_loop_interactive() -> None:
    """
    Точка входа из main.py:
    - даёт выбрать snapshot
    - запускает training-loop на нём.
    """
    working_tf = SETTINGS.market.working_timeframe  # например "H1"
    snapshot_path = _choose_snapshot_interactive(
        role=None,
        tf=working_tf,
    )
    if snapshot_path is None:
        return

    run_training_loop(snapshot_path)

def run_training_loop(snapshot_path: Path) -> None:
    print(f"[Training] Загрузка snapshot: {snapshot_path}")
    if not snapshot_path.exists():
        print(f"[Training] Файл не найден: {snapshot_path}")
        return

    # 1) читаем snapshot (чтобы взять start/end)
    df_snap = pd.read_parquet(snapshot_path)
    print(f"[Training] Исходный snapshot shape: {df_snap.shape}")

    # Извлекаем времена
    if "time" in df_snap.columns:
        times = pd.to_datetime(df_snap["time"])
    elif isinstance(df_snap.index, pd.DatetimeIndex):
        times = df_snap.index.to_series()
    else:
        raise ValueError("Snapshot должен иметь колонку 'time' или DatetimeIndex")

    # Принудительно делаем наивными (без таймзоны)
    times = times.dt.tz_convert('UTC').dt.tz_localize(None) if hasattr(times,
                                                                       'dt') and times.dt.tz is not None else times
    if hasattr(times, 'tz') and times.tz is not None:
        times = times.tz_convert('UTC').tz_localize(None)

    snap_start = times.min()
    snap_end = times.max()

    # Приводим к наивному формату (на всякий случай)
    snap_start = pd.Timestamp(snap_start).tz_localize(None)
    snap_end = pd.Timestamp(snap_end).tz_localize(None)

    # 2) FeatureEngine + профиль
    profile_name = _select_feature_profile()
    engine = FeatureEngine(SETTINGS.features, profile_name=profile_name)
    print(f"[Training] Используем профиль фич: {profile_name}")

    # 3) собираем TF из settings.yaml (например H1/H4/D1)
    required_tfs = _collect_required_tfs(SETTINGS)
    max_tf_minutes = max(_tf_to_minutes(tf) for tf in required_tfs)

    # 4) считаем base warmup в барах рабочего TF (как FeatureEngine уже умеет)
    base_warmup_bars = int(engine._compute_warmup_bars())

    # 5) переводим warmup в "минуты master M1" с учётом самого старшего TF + margin
    warmup_minutes = max(base_warmup_bars * _tf_to_minutes(tf) for tf in required_tfs)
    margin_minutes = max_tf_minutes
    master_start_raw = snap_start - timedelta(minutes=(warmup_minutes + margin_minutes))
    master_end_raw = snap_end

    # Приводим к наивному (без таймзоны)
    master_start = pd.Timestamp(master_start_raw).tz_localize(None)
    master_end = pd.Timestamp(master_end_raw).tz_localize(None)

    # 6) читаем master M1 (с расширением назад)
    master_fname = SETTINGS.paths.master_pattern.format(symbol=SETTINGS.market.symbol, tf="M1")
    master_path = MASTER_DIR / master_fname
    df_m1 = pd.read_parquet(master_path)
    if "time" not in df_m1.columns:
        raise ValueError("Master parquet must contain 'time' column")
    df_m1["time"] = pd.to_datetime(df_m1["time"]).dt.tz_localize(None)  # наивный
    df_m1 = df_m1[(df_m1["time"] >= master_start) & (df_m1["time"] <= master_end)].copy()

    # 7) ресемплим M1 -> working_tf на расширенном диапазоне
    df_work = _resample_m1_to_tf(df_m1, SETTINGS.market.working_timeframe)

    # Сохраняем колонку time и устанавливаем её как индекс
    if "time" not in df_work.columns:
        raise ValueError("После ресэмплинга отсутствует колонка 'time'")
    df_work["time"] = pd.to_datetime(df_work["time"])  # на всякий случай
    df_work = df_work.set_index("time").sort_index()

    # 8) считаем фичи БЕЗ drop_warmup
    df_feat_ext = engine.enrich(df_work, drop_warmup=False)

    # 9) обрезка по датам — time теперь индекс, поэтому фильтруем по index
    df_feat = df_feat_ext[
        (df_feat_ext.index >= snap_start) &
        (df_feat_ext.index <= snap_end)
    ].copy()
    df_feat = df_feat.reset_index()  # сохраняем time как колонку, без drop=True
    print(f"[Training] Результирующий dataframe shape: {df_feat.shape}")

    # --- DEBUG EXPORT (Murrey + ST values only) ---
    N = 500

    cols = ["time", "open", "high", "low", "close"]  # ← time первым

    # SuperTrend values only
    for c in ["st_H1", "st_H4", "st_D1"]:
        if c in df_feat.columns:
            cols.append(c)

    # Murrey levels
    tfs = ["H1", "H4", "D1"]
    mur_levels = [f"mur_{i}_8" for i in range(0, 9)] + ["mur_-2_8", "mur_-1_8", "mur_9_8", "mur_10_8"]
    for tf in tfs:
        for lvl in mur_levels:
            col = f"{lvl}_{tf}"
            if col in df_feat.columns:
                cols.append(col)

    # Murrey derived
    mur_extra = ["mur_zone", "mur_pos_in_zone", "mur_nearest_idx",
                 "mur_dist_close_to_nearest", "mur_dist_close_to_0_8",
                 "mur_dist_close_to_4_8", "mur_dist_close_to_8_8"]
    for tf in tfs:
        for x in mur_extra:
            col = f"{x}_{tf}"
            if col in df_feat.columns:
                cols.append(col)

    out_df = df_feat[cols].tail(N).copy()
    out_df.to_csv("debug_murrey_supertrend_compare.csv", index=False)
    print("[Training] Exported:", "debug_murrey_supertrend_compare.csv", "rows:", len(out_df))

    # последние 3 бара с базовыми полями
    print("\n[Training] Последние 3 бара окна (с базовыми полями):")
    cols_base = [c for c in ("time", "open", "high", "low", "close") if c in df_feat.columns]
    print(df_feat[cols_base].tail(5))

    # формируем вектор фич по последней строке
    feature_cols = [c for c in df_feat.columns if c not in cols_base]
    names = feature_cols
    values = df_feat.iloc[-1][feature_cols].to_numpy(dtype=float)

    feat_dict = dict(zip(names, values))

    print(f"\n[Training] feature_dim = {values.shape[0]}")
    print("[Training] Вектор фич (name = value из последнего бара):")
    for name in names:
        print(f"  {name:15s} = {feat_dict[name]: .6f}")

    print("\n[Training] Черновой training-loop завершён (фичи считаются успешно).")

def run_validation_loop() -> None:
    """
    Заглушка под будущую валидацию.
    """
    print("[Training] run_validation_loop(): ещё не реализовано (STUB)")
