# RL_PyTorch/market_data/market_data_layer.py
from .builders.master_validator import validate_master_history_impl
from config.settings import SETTINGS
from market_data.builders.snapshot_builder import (
    create_snapshot_from_master_impl,
    list_snapshots_impl,
    validate_snapshot_impl,
    delete_snapshot_impl,
)

from .builders.master_builder import (
    update_master_from_live_logs_impl,
    build_master_from_external_impl,
)
from market_data.downloaders.dukascopy_m1_downloader import download_history_from_settings as download_dukascopy_history_from_settings

# --------- History / Master API ---------
def update_master_from_live_logs() -> None:
    print("[History] Обновление master-истории из live-логов (STUB)")
    update_master_from_live_logs_impl()

def validate_master_history() -> None:
    print("[History] Проверка целостности master-истории (STUB)")
    validate_master_history_impl()

def download_history() -> None:
    src = getattr(SETTINGS, "data_source", "dukascopy")
    print(f"[History] Загрузка истории M1 по настройкам (source={src})")
    if src == "dukascopy":
        download_dukascopy_history_from_settings()
    elif src == "mt5":
        print("[MT5] Пока не реализовано: загрузка истории из MT5 в external (будет через ticks→M1).")
        # TODO: подключим, когда добавим downloader для MT5 ticks
    else:
        print(f"[History] Unknown data_source={src}. Supported: dukascopy|mt5")

def build_master_from_external() -> None:
    print("[History] Сборка master-истории из external (Dukascopy → master)")
    src = getattr(SETTINGS, "data_source", "dukascopy")
    print(f"[History] Сборка master-истории из external → master (source={src})")
    build_master_from_external_impl()


# --------- Snapshot API ---------

# --------- Snapshot API ---------

def create_snapshot_from_master(start_date: str, end_date: str, role: str) -> None:
    """
    Обёртка над create_snapshot_from_master_impl для вызова из main.py.
    Даты и роль уже выбраны в snapshot_menu() (main.py).
    Здесь только выбираем TF и передаём всё в билдера.
    """
    default_tf = getattr(SETTINGS.market, "working_timeframe", SETTINGS.market.timeframe)
    tf_in = input(f"Таймфрейм снапшота (Enter = {default_tf}): ").strip().upper()
    tf = tf_in or default_tf

    print(f"[market_data_layer] Создаём snapshot role={role}, tf={tf}, {start_date}..{end_date}")
    create_snapshot_from_master_impl(start_date, end_date, role, tf=tf)


def list_snapshots() -> None:
    list_snapshots_impl()


def validate_snapshot() -> None:
    validate_snapshot_impl()


def delete_snapshot() -> None:
    delete_snapshot_impl()



