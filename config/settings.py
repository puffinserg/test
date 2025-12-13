
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

@dataclass
class SuperTrendSettings:
    enabled: bool = True
    atr_period: int = 5
    multiplier: float = 3.0
    cci_period: int = 50
    cci_price: str = "typical"
    tfs: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    outputs: Dict[str, Any] = field(default_factory=dict)  # пока как сырой dict

@dataclass
class MurreySettings:
    enabled: bool = True
    period_bars: int = 64
    include_extremes: bool = True
    tfs: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    outputs: Dict[str, Any] = field(default_factory=dict)

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
    use_supertrend: bool = True
    use_murrey: bool = True

    # оверрайды базовых фич
    atr_period: Optional[int] = None
    vol_period: Optional[int] = None
    short_periods: Optional[List[int]] = None
    ret_long_period: Optional[int] = None
    spread_period: Optional[int] = None
    supertrend_atr_period: Optional[int] = None
    supertrend_multiplier: Optional[float] = None
    supertrend_cci_period: Optional[int] = None

    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeaturesSettings:
    window_size: int = 128

    atr: AtrSettings = field(default_factory=AtrSettings)
    volatility: VolatilitySettings = field(default_factory=VolatilitySettings)
    returns: ReturnsSettings = field(default_factory=ReturnsSettings)
    spread: SpreadSettings = field(default_factory=SpreadSettings)
    supertrend: SuperTrendSettings = field(default_factory=SuperTrendSettings)
    murrey: MurreySettings = field(default_factory=MurreySettings)

    pipeline: List[str] = field(default_factory=list)

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
    # --- data source switching ---
    data_source: str = "dukascopy"  # "dukascopy" | "mt5"
    data_layout: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, Any] = field(default_factory=dict)

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

    # --- data source switching ---
    data_source = str(raw.get("data_source", "dukascopy")).strip().lower()
    data_layout = raw.get("data_layout", {}) or {}
    sources = raw.get("sources", {}) or {}

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
    st_raw = feat_raw.get("supertrend", {}) or {}
    supertrend = SuperTrendSettings(
        enabled=st_raw.get("enabled", True),
        atr_period=int(st_raw.get("atr_period", 5)),
        multiplier=float(st_raw.get("multiplier", 3.0)),
        cci_period=int(st_raw.get("cci_period", 50)),
        cci_price=st_raw.get("cci_price", "typical"),
        tfs=list(st_raw.get("tfs", ["H1", "H4", "D1"])),
        outputs=st_raw.get("outputs", {}) or {},
    )

    mur_raw = feat_raw.get("murrey", {}) or {}
    murrey = MurreySettings(
        enabled=mur_raw.get("enabled", True),
        period_bars=int(mur_raw.get("period_bars", 64)),
        include_extremes=bool(mur_raw.get("include_extremes", True)),
        tfs=list(mur_raw.get("tfs", ["H1", "H4", "D1"])),
        outputs=mur_raw.get("outputs", {}) or {},
    )

    pipeline = list(feat_raw.get("pipeline", []))

    default_profile = feat_raw.get("default_profile", "name1")
    profiles_raw = feat_raw.get("profiles", {}) or {}
    profiles: Dict[str, FeatureProfileSettings] = {}

    for name, cfg in profiles_raw.items():
        cfg = cfg or {}

        known_keys = {
            "use_atr", "use_volatility", "use_returns", "use_spread",
            "atr_period", "vol_period", "short_periods", "ret_long_period", "spread_period",
            "supertrend_atr_period", "supertrend_multiplier", "supertrend_cci_period", "use_supertrend", "use_murrey",
        }

        # всё остальное сохраняем как “raw overrides” (например murrey: {outputs: ...}, supertrend: {outputs: ...})
        overrides = {k: v for k, v in cfg.items() if k not in known_keys}

        profiles[name] = FeatureProfileSettings(
            use_atr=cfg.get("use_atr", True),
            use_volatility=cfg.get("use_volatility", True),
            use_returns=cfg.get("use_returns", True),
            use_spread=cfg.get("use_spread", True),
            use_supertrend=cfg.get("use_supertrend", True),
            use_murrey=cfg.get("use_murrey", True),
            atr_period=cfg.get("atr_period"),
            vol_period=cfg.get("vol_period"),
            short_periods=cfg.get("short_periods"),
            ret_long_period=cfg.get("ret_long_period"),
            spread_period=cfg.get("spread_period"),
            supertrend_atr_period=cfg.get("supertrend_atr_period"),
            supertrend_multiplier=cfg.get("supertrend_multiplier"),
            supertrend_cci_period=cfg.get("supertrend_cci_period"),
            overrides=overrides,
        )

    features = FeaturesSettings(
        window_size=int(feat_raw.get("window_size", 128)),
        atr=atr,
        volatility=vol,
        returns=ret,
        spread=spr,
        supertrend=supertrend,
        murrey=murrey,
        pipeline=pipeline,
        default_profile=default_profile,
        profiles=profiles,
    )

    return Settings(
        market=market,
        paths=paths,
        dukascopy=dukascopy,
        features=features,
        data_source=data_source,
        data_layout = data_layout,
        sources = sources,
    )

# Глобальный объект настроек
SETTINGS: Settings = load_settings()
