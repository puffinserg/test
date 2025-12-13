# features/feature_engine.py

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, fields
from typing import Sequence, Callable, Dict, Optional, List, Any

import copy
import numpy as np
import pandas as pd

from config.settings import SETTINGS, FeaturesSettings, FeatureProfileSettings
from features.mtf_utils import resample_ohlc, align_higher_tf_to_working
from features.indicators import compute_murrey_grid
from features.indicators import (
    compute_supertrend_and_cci,
    compute_atr,
    compute_returns,
    compute_volatility,
    compute_spread_stats,
)

# ---------- Feature registry infrastructure ----------
FeatureFunc = Callable[[pd.DataFrame, "FeatureContext"], None]
LookbackFunc = Callable[["FeatureContext"], int]


@dataclass
class FeatureMeta:
    name: str
    default_enabled: bool = True
    lookback_fn: Optional[LookbackFunc] = None


@dataclass
class FeatureContext:
    """
    Контекст, передаваемый в каждую фичу:
    - engine: сам FeatureEngine
    - cfg: глобальные настройки features (FeaturesSettings)
    - profile: активный профиль фич (FeatureProfileSettings)
    """
    engine: "FeatureEngine"
    cfg: FeaturesSettings
    profile: FeatureProfileSettings


FEATURE_REGISTRY: Dict[str, FeatureFunc] = {}
FEATURE_META: Dict[str, FeatureMeta] = {}

TF_TO_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def _tf_factor(tf: str, base_tf: str) -> float:
    """
    Во сколько раз tf «дольше» base_tf.
    Например: H4 vs H1 => 4, D1 vs H1 => 24.
    """
    base = TF_TO_MINUTES.get(base_tf)
    cur = TF_TO_MINUTES.get(tf)
    if base is None or cur is None:
        return 1.0
    return cur / base


def _sort_tfs_by_tf_order(tfs: list[str]) -> list[str]:
    """
    Сортировка таймфреймов от младшего к старшему (M1, M5, ..., H1, H4, D1)
    по TF_TO_MINUTES.
    """
    return sorted(
        set(tfs),
        key=lambda tf: TF_TO_MINUTES.get(tf, 10**9),
    )


def register_feature(
    name: str,
    *,
    default_enabled: bool = True,
    lookback_fn: Optional[LookbackFunc] = None,
) -> Callable[[FeatureFunc], FeatureFunc]:
    """
    Декоратор для регистрации фичи.
    name — ключ фичи (используется в YAML/pipeline).
    """
    def decorator(func: FeatureFunc) -> FeatureFunc:
        FEATURE_REGISTRY[name] = func
        FEATURE_META[name] = FeatureMeta(
            name=name,
            default_enabled=default_enabled,
            lookback_fn=lookback_fn,
        )
        return func

    return decorator


# ---------- Lookback helpers ----------
def supertrend_lookback(ctx: FeatureContext) -> int:
    """
    Сколько баров рабочего TF нужно, чтобы SuperTrend был «устоявшимся»
    на ВСЕХ таймфреймах из cfg.supertrend.tfs.

    База: max(atr_period, cci_period) * 6 на TF SuperTrend.
    Дальше умножаем на max-множитель между рабочим TF и старшими TF.
    """
    engine = ctx.engine
    cfg_st = ctx.cfg.supertrend

    atr_p = engine._eff_st_atr_period()
    cci_p = engine._eff_st_cci_period()

    base_lookback = max(atr_p, cci_p) * 6

    working_tf = SETTINGS.market.working_timeframe
    tfs = cfg_st.tfs or [working_tf]
    factors = [_tf_factor(tf, working_tf) for tf in tfs]
    max_factor = max(factors) if factors else 1.0

    lookback = int(base_lookback * max_factor)
    return max(1, lookback)


def murrey_lookback(ctx: FeatureContext) -> int:
    """
    Сколько баров рабочего TF нужно, чтобы уровни Мюррея стабилизировались
    на всех таймфреймах из cfg.murrey.tfs.
    """
    engine = ctx.engine
    cfg_m = ctx.cfg.murrey
    period_bars = cfg_m.period_bars

    # коэффициент запаса для сетки (можно вынести в YAML позже)
    k = 2.0
    base_lookback = int(period_bars * k)

    working_tf = SETTINGS.market.working_timeframe
    tfs = cfg_m.tfs or [working_tf]
    factors = [_tf_factor(tf, working_tf) for tf in tfs]
    max_factor = max(factors) if factors else 1.0

    lookback = int(base_lookback * max_factor)
    return max(1, lookback)


# ---------- Deep-merge of profile overrides ----------
def _deep_merge_into_dataclass(dst_obj: Any, patch: Dict[str, Any]) -> None:
    """
    Рекурсивно накатывает patch (dict) на dataclass/dict-структуры внутри dst_obj.
    - dataclass: патчит поля по имени
    - dict: deep merge по ключам
    - list/scalar: замена целиком
    """
    if patch is None:
        return
    if not isinstance(patch, dict):
        return

    if is_dataclass(dst_obj):
        field_map = {f.name: f for f in fields(dst_obj)}
        for k, v in patch.items():
            if k not in field_map:
                # неизвестные ключи игнорируем (можно логировать при verbose)
                continue

            cur = getattr(dst_obj, k)

            if is_dataclass(cur) and isinstance(v, dict):
                _deep_merge_into_dataclass(cur, v)
            elif isinstance(cur, dict) and isinstance(v, dict):
                _deep_merge_into_dict(cur, v)
            else:
                setattr(dst_obj, k, copy.deepcopy(v))
        return

    if isinstance(dst_obj, dict):
        _deep_merge_into_dict(dst_obj, patch)


def _deep_merge_into_dict(dst: Dict[str, Any], patch: Dict[str, Any]) -> None:
    for k, v in patch.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge_into_dict(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def apply_profile_feature_overrides(cfg: Any, profile: Any) -> None:
    """
    Накатывает profile.overrides на cfg (FeaturesSettings).

    Ожидается overrides вида:
      { "murrey": {...}, "supertrend": {...}, ... }
    """
    overrides = getattr(profile, "overrides", None)
    if not overrides:
        return
    _deep_merge_into_dataclass(cfg, overrides)


# ---------- Вспомогательная обёртка над SETTINGS.features (опционально) ----------
@dataclass
class FeatureEngineConfig:
    """
    Простейшая обёртка над SETTINGS.features, если вдруг понадобится использовать FeatureEngine
    без прямой ссылки на SETTINGS.
    """
    window_size: int
    atr_period: int
    vol_period: int
    short_periods: Sequence[int]
    ret_long_period: int
    spread_period: int

    @classmethod
    def from_settings(cls) -> "FeatureEngineConfig":
        f = SETTINGS.features
        return cls(
            window_size=f.window_size,
            atr_period=f.atr.atr_period,
            vol_period=f.volatility.vol_period,
            short_periods=list(f.returns.short_periods),
            ret_long_period=f.returns.ret_long_period,
            spread_period=f.spread.spread_period,
        )


# ---------- Основной класс FeatureEngine ----------
class FeatureEngine:
    """
    Счётчик фич поверх "чистого" snapshot’а:
    - ATR
    - ret_*
    - volatility
    - spread_mean, spread_std, spread_over_atr
    - supertrend (+ CCI) MTF
    - murrey MTF
    """

    def __init__(self, cfg: FeaturesSettings, profile_name: Optional[str] = None):
        self.cfg: FeaturesSettings = cfg
        self.profile_name = profile_name or cfg.default_profile
        self.profile: FeatureProfileSettings = cfg.profiles.get(
            self.profile_name, FeatureProfileSettings()
        )

        # ✅ КЛЮЧЕВОЕ: применяем overrides профиля (murrey.outputs.*, supertrend.outputs.*, ...)
        apply_profile_feature_overrides(self.cfg, self.profile)

    # --------- эффективные параметры ---------
    def _eff_atr_period(self) -> int:
        return self.profile.atr_period or self.cfg.atr.atr_period

    def _eff_vol_period(self) -> int:
        return self.profile.vol_period or self.cfg.volatility.vol_period

    def _eff_ret_long_period(self) -> int:
        return self.profile.ret_long_period or self.cfg.returns.ret_long_period

    def _eff_short_periods(self) -> List[int]:
        return self.profile.short_periods or list(self.cfg.returns.short_periods)

    def _eff_spread_period(self) -> int:
        return self.profile.spread_period or self.cfg.spread.spread_period

    def _eff_st_atr_period(self) -> int:
        return self.profile.supertrend_atr_period or self.cfg.supertrend.atr_period

    def _eff_st_multiplier(self) -> float:
        return self.profile.supertrend_multiplier or self.cfg.supertrend.multiplier

    def _eff_st_cci_period(self) -> int:
        return self.profile.supertrend_cci_period or self.cfg.supertrend.cci_period

    # --------- lookback / warmup ----------
    def _compute_warmup_bars(self) -> int:
        """
        Расчёт warmup:
        1) базовый механизм (на основе периодов),
        2) плюс минимальный lookback, требуемый зарегистрированными фичами.
        """
        atr_p = self._eff_atr_period()
        vol_p = self._eff_vol_period()
        ret_long_p = self._eff_ret_long_period()
        shorts = self._eff_short_periods()
        short_max = max(shorts) if shorts else 0
        spread_p = self._eff_spread_period()

        base_period = max(atr_p, vol_p, ret_long_p, spread_p, short_max)
        warmup_old = max(self.cfg.window_size, 4 * base_period)

        ctx = FeatureContext(engine=self, cfg=self.cfg, profile=self.profile)
        pipeline = self._resolve_feature_pipeline()

        registry_lb = 0
        for name in pipeline:
            meta = FEATURE_META.get(name)
            if meta and meta.lookback_fn:
                try:
                    lb = meta.lookback_fn(ctx)
                    if lb is not None:
                        registry_lb = max(registry_lb, int(lb))
                except Exception as e:
                    print(f"[feature_engine] WARNING: lookback_fn for '{name}' failed: {e}")

        warmup = max(warmup_old, registry_lb)
        return warmup

    # --------- порядок фич в pipeline ----------
    def _resolve_feature_pipeline(self) -> List[str]:
        """
        Определяет порядок применения фич:
        1) если в cfg.pipeline задан список — используем его;
        2) иначе используем фоллбек-порядок по use_*.
        """
        pipeline = getattr(self.cfg, "pipeline", None)
        if pipeline:
            return list(pipeline)

        order: List[str] = []
        if getattr(self.profile, "use_atr", True):
            order.append("atr")
        if getattr(self.profile, "use_returns", True):
            order.append("returns")
        if getattr(self.profile, "use_volatility", True):
            order.append("volatility")
        if getattr(self.profile, "use_spread", True):
            order.append("spread")
        return order

    def _feature_enabled_by_profile(self, name: str) -> bool:
        """
        Связывает названия фич в реестре с полями профиля (для базовых фич).
        """
        if name == "atr":
            return getattr(self.profile, "use_atr", True)
        if name == "returns":
            return getattr(self.profile, "use_returns", True)
        if name == "volatility":
            return getattr(self.profile, "use_volatility", True)
        if name == "spread":
            return getattr(self.profile, "use_spread", True)
        if name == "supertrend":
            return getattr(self.profile, "use_supertrend", True)
        if name == "murrey":
            return getattr(self.profile, "use_murrey", True)

        meta = FEATURE_META.get(name)
        return meta.default_enabled if meta is not None else True

    # --------- основной метод ---------
    def enrich(
        self,
        df: pd.DataFrame,
        drop_warmup: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        На вход: чистый snapshot (time, OHLC, spread, volume).
        На выход: df с фичами, при необходимости с отрезанным warmup.
        """
        df = df.copy()

        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)

        if verbose:
            print("[feature_engine] Enriching dataframe with features...")

        ctx = FeatureContext(engine=self, cfg=self.cfg, profile=self.profile)

        pipeline = self._resolve_feature_pipeline()
        if verbose:
            print(f"[feature_engine] Feature pipeline: {pipeline}")

        for name in pipeline:
            func = FEATURE_REGISTRY.get(name)
            if func is None:
                if verbose:
                    print(f"[feature_engine] WARNING: feature '{name}' not registered")
                continue

            if not self._feature_enabled_by_profile(name):
                if verbose:
                    print(f"[feature_engine] Feature '{name}' disabled by profile")
                continue

            if verbose:
                print(f"[feature_engine] Applying feature '{name}'")
            func(df, ctx)

        if drop_warmup:
            warmup = self._compute_warmup_bars()
            if verbose:
                print(f"[feature_engine] Dropping warmup: first {warmup} rows")
            df = df.iloc[warmup:].reset_index(drop=True)

        feature_cols = [
            c for c in df.columns
            if c not in ("time", "open", "high", "low", "close", "tick_volume", "real_volume")
        ]
        if feature_cols:
            nan_rows = df[feature_cols].isna().any(axis=1).sum()
            if verbose and nan_rows > 0:
                print(
                    f"[feature_engine] WARNING: {nan_rows} rows still contain NaN "
                    f"in feature columns after warmup"
                )

        return df


# ---------- Реализации базовых фич ----------
@register_feature(
    "atr",
    default_enabled=True,
    lookback_fn=lambda ctx: max(1, ctx.engine._eff_atr_period()) * 4,
)
def feature_atr(df: pd.DataFrame, ctx: FeatureContext) -> None:
    profile = ctx.profile
    engine = ctx.engine
    if not getattr(profile, "use_atr", True):
        return
    if not {"high", "low", "close"}.issubset(df.columns):
        print("[feature_engine] WARNING: cannot compute ATR – OHLC missing")
        return
    period = engine._eff_atr_period()
    df["atr"] = compute_atr(df, period)


@register_feature(
    "returns",
    default_enabled=True,
    lookback_fn=lambda ctx: max(
        [ctx.engine._eff_ret_long_period()] + (ctx.engine._eff_short_periods() or [0])
    ) * 2,
)
def feature_returns(df: pd.DataFrame, ctx: FeatureContext) -> None:
    profile = ctx.profile
    engine = ctx.engine
    if not getattr(profile, "use_returns", True):
        return
    if "close" not in df.columns:
        print("[feature_engine] WARNING: cannot compute returns – 'close' missing")
        return

    close = df["close"]
    shorts = engine._eff_short_periods()
    long_p = engine._eff_ret_long_period()

    returns_dict = compute_returns(close, shorts, long_p)
    for col, series in returns_dict.items():
        df[col] = series


@register_feature(
    "volatility",
    default_enabled=True,
    lookback_fn=lambda ctx: max(1, ctx.engine._eff_vol_period()) * 2,
)
def feature_volatility(df: pd.DataFrame, ctx: FeatureContext) -> None:
    profile = ctx.profile
    engine = ctx.engine
    if not getattr(profile, "use_volatility", True):
        return
    if "close" not in df.columns:
        print("[feature_engine] WARNING: cannot compute volatility – 'close' missing")
        return

    period = engine._eff_vol_period()
    df["volatility"] = compute_volatility(df["close"], period)


@register_feature(
    "spread",
    default_enabled=True,
    lookback_fn=lambda ctx: max(1, ctx.engine._eff_spread_period()) * 2,
)
def feature_spread(df: pd.DataFrame, ctx: FeatureContext) -> None:
    profile = ctx.profile
    engine = ctx.engine
    if not getattr(profile, "use_spread", True):
        return

    if "spread" not in df.columns:
        print("[feature_engine] WARNING: cannot compute spread stats – 'spread' missing")
        return

    period = engine._eff_spread_period()
    stats = compute_spread_stats(df["spread"], df.get("atr"), period)
    for col, series in stats.items():
        df[col] = series


# ---------- SuperTrend (+CCI) ----------
@register_feature("supertrend", default_enabled=True, lookback_fn=supertrend_lookback)
def feature_supertrend(df: pd.DataFrame, ctx: FeatureContext) -> None:
    engine = ctx.engine
    cfg_st = ctx.cfg.supertrend
    if not cfg_st.enabled:
        return

    if not {"open", "high", "low", "close"}.issubset(df.columns):
        print("[feature_engine] WARNING: cannot compute supertrend – OHLC missing")
        return

    working_tf = SETTINGS.market.working_timeframe
    base_atr = engine._eff_st_atr_period()
    mult = engine._eff_st_multiplier()
    cci_p = engine._eff_st_cci_period()
    cci_price = cfg_st.cci_price

    outputs = cfg_st.outputs or {}

    def _emit_supertrend_cols(st_df: pd.DataFrame, tf_suffix: str) -> None:
        # st_df содержит колонки, которые вернул compute_supertrend_and_cci()
        for col in st_df.columns:
            df[f"{col}_{tf_suffix}"] = st_df[col]

    tfs = _sort_tfs_by_tf_order(cfg_st.tfs or [working_tf])

    # рабочий TF
    st0 = compute_supertrend_and_cci(
        df=df,
        atr_period=base_atr,
        multiplier=mult,
        cci_period=cci_p,
        cci_price=cci_price,
        outputs=outputs,
    )
    _emit_supertrend_cols(st0, working_tf)

    # старшие TF
    for tf in tfs:
        if tf == working_tf:
            continue
        htf = resample_ohlc(df, tf)
        st_htf = compute_supertrend_and_cci(
            df=htf,
            atr_period=base_atr,
            multiplier=mult,
            cci_period=cci_p,
            cci_price=cci_price,
            outputs=outputs,
        )
        st_aligned = align_higher_tf_to_working(st_htf, df, how="backward")
        _emit_supertrend_cols(st_aligned, tf)


# ---------- Murrey ----------
@register_feature("murrey", default_enabled=True, lookback_fn=murrey_lookback)
def feature_murrey(df: pd.DataFrame, ctx: FeatureContext) -> None:
    cfg_m = ctx.cfg.murrey
    if not cfg_m.enabled:
        return

    if not {"high", "low", "close"}.issubset(df.columns):
        print("[feature_engine] WARNING: cannot compute murrey – OHLC missing")
        return

    working_tf = SETTINGS.market.working_timeframe
    period_bars = int(cfg_m.period_bars)
    include_extremes = bool(getattr(cfg_m, "include_extremes", True))
    outputs = cfg_m.outputs or {}

    def _emit_murrey(mdf: pd.DataFrame, tf_suffix: str) -> None:
        for col in mdf.columns:
            df[f"{col}_{tf_suffix}"] = mdf[col]

    tfs = _sort_tfs_by_tf_order(cfg_m.tfs or [working_tf])

    m0 = compute_murrey_grid(
        df=df,
        period_bars=period_bars,
        include_extremes=include_extremes,
        outputs=outputs,
    )
    _emit_murrey(m0, working_tf)

    for tf in tfs:
        if tf == working_tf:
            continue
        htf = resample_ohlc(df, tf)
        m_htf = compute_murrey_grid(
            df=htf,
            period_bars=period_bars,
            include_extremes=include_extremes,
            outputs=outputs,
        )
        m_aligned = align_higher_tf_to_working(m_htf, df, how="backward")
        _emit_murrey(m_aligned, tf)
