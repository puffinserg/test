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
from typing import Optional, List

import pandas as pd

from config.settings import SETTINGS
from features.feature_engine import FeatureEngine


BASE_DIR = Path(__file__).resolve().parents[2]  # .../RL_PyTorch
SNAPSHOT_DIR = BASE_DIR / "data" / "snapshots"

# ---------- вспомогательные функции ----------

from typing import Optional, List, Dict
...

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
        print(f"  {i}. {path.name}")

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

# ---------- публичные функции ----------

def run_training_loop_interactive() -> None:
    """
    Точка входа из main.py:
    - даёт выбрать snapshot
    - запускает training-loop на нём.
    """
    working_tf = SETTINGS.market.working_timeframe  # например "H1"
    snapshot_path = _choose_snapshot_interactive(
        role="train",
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

    df = pd.read_parquet(snapshot_path)
    print(f"[Training] Исходный snapshot shape: {df.shape}")

    profile_name = _select_feature_profile()
    engine = FeatureEngine(SETTINGS.features, profile_name=profile_name)
    print(f"[Training] Используем профиль фич: {profile_name}")

    df_feat = engine.enrich(df, drop_warmup=True)
    print(f"[Training] Результирующий dataframe shape: {df_feat.shape}")

    # последние 3 бара с базовыми полями
    print("\n[Training] Последние 3 бара окна (с базовыми полями):")
    cols_base = [c for c in ("time", "open", "high", "low", "close", "st_H1", "st_H4", "st_D1") if c in df_feat.columns]
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
