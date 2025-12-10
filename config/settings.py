
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
SETTINGS_FILE = Path(__file__).with_name("settings.yaml")


# ---------- Секции настроек ----------

@dataclass
class AtrSettings:
    atr_period: int = 14

@dataclass
class VolatilitySettings:
    vol_period: int = 20

@dataclass
class ReturnsSettings:
    short_periods: List[int] = field(default_factory=lambda: [1, 5])
    ret_long_period: int = 5

@dataclass
class SpreadSettings:
    spread_period: int = 20

# --- Профиль фич ---

@dataclass
class FeatureProfileSettings:
    """
    Профиль фич (name1, name2, ...):
    - какие блоки использовать;
    - при желании – локальные переопределения периодов.
    """
    use_atr: bool = True
    use_volatility: bool = True
    use_returns: bool = True
    use_spread: bool = True

    atr_period: Optional[int] = None
    vol_period: Optional[int] = None
    short_periods: Optional[List[int]] = None
    ret_long_period: Optional[int] = None
    spread_period: Optional[int] = None

@dataclass
class FeaturesSettings:
    window_size: int = 128

    atr: AtrSettings = field(default_factory=AtrSettings)
    volatility: VolatilitySettings = field(default_factory=VolatilitySettings)
    returns: ReturnsSettings = field(default_factory=ReturnsSettings)
    spread: SpreadSettings = field(default_factory=SpreadSettings)

    default_profile: str = "name1"
    profiles: Dict[str, FeatureProfileSettings] = field(default_factory=dict)

# --- Корневой SETTINGS ---

@dataclass
class MarketSettings:
    symbol: str = "EURUSD"
    timeframe: str = "M1"
    working_timeframe: str = "H1"
    n_bars_master: int = 70000

@dataclass
class PathsSettings:
    master_pattern: str = "{symbol}_{tf}_master.parquet"
    snapshot_pattern: str = "{symbol}_{tf}_{role}_snapshot_{start}_{end}.parquet"

@dataclass
class DukascopySettings:
    symbol: Optional[str] = None
    start_year: int = 2010
    years: int = 15
    external_dir: str = "data/external"
    price_divisor: int = 100000

@dataclass
class Settings:
    market: MarketSettings = field(default_factory=MarketSettings)
    paths: PathsSettings = field(default_factory=PathsSettings)
    dukascopy: DukascopySettings = field(default_factory=DukascopySettings)
    features: FeaturesSettings = field(default_factory=FeaturesSettings)

# ---------- Загрузка YAML ----------

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_settings(path: Path | None = None) -> Settings:
    if path is None:
        path = SETTINGS_FILE

    raw = _load_yaml(path)

    # --- market ---
    m = raw.get("market", {}) or {}
    market = MarketSettings(
        symbol=m.get("symbol", "EURUSD"),
        timeframe=m.get("timeframe", "M1"),
        working_timeframe=m.get(
            "working_timeframe",
            m.get("timeframe", "M1"),
        ),
        n_bars_master=int(m.get("n_bars_master", 70_000)),
    )

    # --- paths ---
    p = raw.get("paths", {}) or {}
    paths = PathsSettings(
        master_pattern=p.get("master_pattern", "{symbol}_{tf}_master.parquet"),
        snapshot_pattern=p.get(
            "snapshot_pattern",
            "{symbol}_{tf}_{role}_snapshot_{start}_{end}.parquet",
        ),
    )

    # --- dukascopy ---
    d = raw.get("dukascopy", {}) or {}
    dukascopy = DukascopySettings(
        symbol=d.get("symbol"),
        start_year=int(d.get("start_year", 2010)),
        years=int(d.get("years", 15)),
        external_dir=d.get("external_dir", "data/external"),
        price_divisor=int(d.get("price_divisor", 100_000)),
    )

    # --- features ---
    feat_raw = raw.get("features", {}) or {}

    atr = AtrSettings(**(feat_raw.get("atr", {}) or {}))
    vol = VolatilitySettings(**(feat_raw.get("volatility", {}) or {}))
    ret = ReturnsSettings(**(feat_raw.get("returns", {}) or {}))
    spr = SpreadSettings(**(feat_raw.get("spread", {}) or {}))

    default_profile = feat_raw.get("default_profile", "name1")
    profiles_raw = feat_raw.get("profiles", {}) or {}

    profiles: Dict[str, FeatureProfileSettings] = {}
    for name, cfg in profiles_raw.items():
        cfg = cfg or {}
        profiles[name] = FeatureProfileSettings(
            use_atr=cfg.get("use_atr", True),
            use_volatility=cfg.get("use_volatility", True),
            use_returns=cfg.get("use_returns", True),
            use_spread=cfg.get("use_spread", True),
            atr_period=cfg.get("atr_period"),
            vol_period=cfg.get("vol_period"),
            short_periods=cfg.get("short_periods"),
            ret_long_period=cfg.get("ret_long_period"),
            spread_period=cfg.get("spread_period"),
        )

    features = FeaturesSettings(
        window_size=int(feat_raw.get("window_size", 128)),
        atr=atr,
        volatility=vol,
        returns=ret,
        spread=spr,
        default_profile=default_profile,
        profiles=profiles,
    )

    return Settings(
        market=market,
        paths=paths,
        dukascopy=dukascopy,
        features=features,
    )

# Глобальный объект настроек
SETTINGS: Settings = load_settings()
