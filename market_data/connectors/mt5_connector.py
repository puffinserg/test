# RL_PyTorch/market_data/connectors/mt5_connector.py
from __future__ import annotations

from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from config.settings import SETTINGS


# Маппинг строкового ТФ из конфига → константы MT5
_TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def get_symbol() -> str:
    return SETTINGS.market.symbol


def get_timeframe() -> int:
    tf_str = SETTINGS.market.timeframe.upper()
    if tf_str not in _TF_MAP:
        raise ValueError(f"Unsupported timeframe in settings.yaml: {tf_str}")
    return _TF_MAP[tf_str]


# ---------- Базовые операции ----------

def init_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("MT5 account_info() returned None")
    print(f"[MT5] Connected to {info.server}, account {info.login}")


def shutdown_mt5() -> None:
    mt5.shutdown()
    print("[MT5] Shutdown")


def download_m1_history(
    symbol: Optional[str] = None,
    n_bars: Optional[int] = None,
    timeframe: Optional[int] = None,
) -> pd.DataFrame:
    """
    Загружает последние n_bars свечей по symbol.

    Если параметры не переданы явно, берём их из SETTINGS.
    """
    if symbol is None:
        symbol = get_symbol()
    if n_bars is None:
        n_bars = SETTINGS.market.n_bars_master
    if timeframe is None:
        timeframe = get_timeframe()

    print(f"[MT5] Loading {n_bars} bars for {symbol} (timeframe={timeframe}) ...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)

    print("[MT5] First rows:")
    print(df.head())
    print("[MT5] Last rows:")
    print(df.tail())

    return df


# ---------- Сервисные функции для меню ----------

def test_mt5_connection() -> None:
    try:
        init_mt5()
    except Exception as e:
        print(f"[Service] MT5 connection FAILED: {e}")
    else:
        print("[Service] MT5 connection OK")
    finally:
        shutdown_mt5()


def show_account_info() -> None:
    try:
        init_mt5()
        info = mt5.account_info()
        if info is None:
            print("[Service] account_info() returned None")
        else:
            print("[Service] Account info:")
            for k, v in info._asdict().items():
                print(f"   {k}: {v}")
    except Exception as e:
        print(f"[Service] Failed to get account info: {e}")
    finally:
        shutdown_mt5()


def list_symbols() -> None:
    try:
        init_mt5()
        symbols = mt5.symbols_get()
        print(f"[Service] Total symbols: {len(symbols)}")
        for s in symbols[:50]:
            print("   ", s.name)
        if len(symbols) > 50:
            print("   ...")
    except Exception as e:
        print(f"[Service] Failed to list symbols: {e}")
    finally:
        shutdown_mt5()
