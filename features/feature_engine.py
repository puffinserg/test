# features/feature_engine.py

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, fields
from typing import Sequence, Callable, Dict, Optional, List, Any
from features.mtf_utils import resample_ohlc, align_higher_tf_to_working
from features.indicators import compute_murrey_grid
from features.indicators import (
    compute_supertrend_and_cci,
    compute_atr,
    compute_returns,
    compute_volatility,
    compute_spread_stats,
)

import numpy as np
import pandas as pd
import copy

from config.settings import SETTINGS, FeaturesSettings, FeatureProfileSettings

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
    Сортировка таймфреймов от младшего к старшему
    (M1, M5, ..., H1, H4, D1) по TF_TO_MINUTES.
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

    # рабочий TF (например, H1)
    from config.settings import SETTINGS
    working_tf = SETTINGS.market.working_timeframe

    tfs = cfg_st.tfs or [working_tf]
    factors = [_tf_factor(tf, working_tf) for tf in tfs]
    max_factor = max(factors) if factors else 1.0

    lookback = int(base_lookback * max_factor)
    # safety: хотя бы 1 бар
    return max(1, lookback)

# в feature_engine.py, рядом с supertrend_lookback

def murrey_lookback(ctx: FeatureContext) -> int:
    """
    Сколько баров рабочего TF нужно, чтобы уровни Мюррея стабилизировались
    на всех таймфреймах из cfg.murrey.tfs.
    """
    engine = ctx.engine
    cfg_m = ctx.cfg.murrey

    period_bars = cfg_m.period_bars  # 64 / 128 / 200 ...
    # коэффициент запаса для сетки: можно будет вынести в YAML, если захотим
    k = 2.0

    base_lookback = int(period_bars * k)

    from config.settings import SETTINGS
    working_tf = SETTINGS.market.working_timeframe

    tfs = cfg_m.tfs or [working_tf]
    factors = [_tf_factor(tf, working_tf) for tf in tfs]
    max_factor = max(factors) if factors else 1.0

    lookback = int(base_lookback * max_factor)
    return max(1, lookback)

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
        # если профилем пришло “не dict”, ничего не делаем (безопасно)
        return

    # dataclass case
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
                # scalar/list/dict replacement
                setattr(dst_obj, k, copy.deepcopy(v))
        return

    # dict case
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
    Накатывает profile.overrides только на cfg.features.*.
    Ожидает, что overrides имеет вид:
      { "murrey": {...}, "supertrend": {...}, ... }
    """
    overrides = getattr(profile, "overrides", None)
    if not overrides:
        return
    # Накатываем только то, что относится к features.*
    _deep_merge_into_dataclass(cfg.features, overrides)

# ---------- Вспомогательная обёртка над SETTINGS.features (опционально) ----------

@dataclass
class FeatureEngineConfig:
    """
    Простейшая обёртка над SETTINGS.features, если вдруг понадобится
    использовать FeatureEngine без прямой ссылки на SETTINGS.
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
    """

    def __init__(self, cfg: FeaturesSettings, profile_name: Optional[str] = None):
        self.cfg: FeaturesSettings = cfg
        self.profile_name = profile_name or cfg.default_profile
        self.profile: FeatureProfileSettings = cfg.profiles.get(
            self.profile_name, FeatureProfileSettings()
        )

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
        # Часть 1: как было раньше — отталкиваемся от периодов
        atr_p = self._eff_atr_period()
        vol_p = self._eff_vol_period()
        ret_long_p = self._eff_ret_long_period()
        shorts = self._eff_short_periods()
        short_max = max(shorts) if shorts else 0
        spread_p = self._eff_spread_period()

        base_period = max(atr_p, vol_p, ret_long_p, spread_p, short_max)
        warmup_old = max(self.cfg.window_size, 4 * base_period)

        # Часть 2: вклад от lookback_fn в реестре
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
        2) иначе используем «наследуемый» порядок (atr → returns → volatility → spread).
        """
        # 1. Если в конфиге есть pipeline — берём его
        pipeline = getattr(self.cfg, "pipeline", None)
        if pipeline:
            return list(pipeline)

        # 2. Фоллбек: старый порядок по use_*
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
        Новые фичи (например, supertrend) можно будет включать/выключать через YAML.
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

        # Для новых фич по умолчанию считаем, что они включены
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

        # Страхуемся, что строки отсортированы по времени
        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)

        if verbose:
            print("[feature_engine] Enriching dataframe with features...")

        # контекст для фичей
        ctx = FeatureContext(engine=self, cfg=self.cfg, profile=self.profile)

        # порядок фич
        pipeline = self._resolve_feature_pipeline()
        if verbose:
            print(f"[feature_engine] Feature pipeline: {pipeline}")

        # применение фич
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

        # отрезаем warmup-зону
        if drop_warmup:
            warmup = self._compute_warmup_bars()
            if verbose:
                print(f"[feature_engine] Dropping warmup: first {warmup} rows")
            df = df.iloc[warmup:].reset_index(drop=True)

        # Проверяем наличие NaN в фичах, но строки НЕ удаляем
        feature_cols = [
            c
            for c in df.columns
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


# ---------- Реализации базовых фич (через реестр) ----------


@register_feature(
    "atr",
    default_enabled=True,
    lookback_fn=lambda ctx: max(1, ctx.engine._eff_atr_period()) * 4,
)
def feature_atr(df: pd.DataFrame, ctx: FeatureContext) -> None:
    """
    ATR на основе стандартного True Range.
    """
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
    """
    ret_p и ret_long_p по стоимости закрытия (close).
    """
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
    """
    Волатильность: rolling std от дневной доходности (ret_1).
    """
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
    """
    Фичи по спреду:
    - spread_mean
    - spread_std
    - spread_over_atr (отношение среднего спреда к ATR)
    """
    profile = ctx.profile
    engine = ctx.engine

    if not getattr(profile, "use_spread", True):
        return
    if "spread" not in df.columns:
        print("[feature_engine] WARNING: cannot compute spread features – 'spread' missing")
        return

    period = engine._eff_spread_period()
    spread = df["spread"]

    atr_series = df["atr"] if "atr" in df.columns else None

    spread_mean, spread_std, spread_over_atr = compute_spread_stats(
        spread=spread,
        period=period,
        atr=atr_series,
    )

    df["spread_mean"] = spread_mean
    df["spread_std"] = spread_std
    df["spread_over_atr"] = spread_over_atr


@register_feature(
    "supertrend",
    default_enabled=False,
    lookback_fn=supertrend_lookback,
)
def feature_supertrend(df: pd.DataFrame, ctx: FeatureContext) -> None:
    """
    SuperTrend + CCI с поддержкой MTF.

    Делает:
    - считает SuperTrend и CCI на рабочем TF (working_timeframe);
    - при наличии старших TF в cfg.supertrend.tfs (H4, D1 и т.п.)
      ресемплит OHLC через mtf_utils.resample_ohlc,
      считает SuperTrend/CCI на старшем TF
      и подтягивает значения на рабочий TF через align_higher_tf_to_working;
    - создаёт колонки в соответствии с settings.yaml -> features.supertrend.outputs:
        st_<TF>, st_dir_<TF>, cci_<TF>, cci_sign_<TF>, <OHLC>_minus_st_<TF>.
    """
    profile = ctx.profile
    engine = ctx.engine
    cfg_st = ctx.cfg.supertrend

    # отключено профилем или глобально
    if not getattr(profile, "use_supertrend", True):
        return
    if not cfg_st.enabled:
        return

    required_cols = {"time", "high", "low", "close", "open"}
    if not required_cols.issubset(df.columns):
        print("[feature_engine] WARNING: cannot compute SuperTrend – OHLC/time missing")
        return

    # Рабочий таймфрейм (например, "H1")
    try:
        working_tf = SETTINGS.market.working_timeframe
    except Exception:
        working_tf = "H1"

    # --- Параметры из профиля / конфигурации ---
    atr_period = engine._eff_st_atr_period()
    multiplier = engine._eff_st_multiplier()
    cci_period = engine._eff_st_cci_period()

    # --- Чтение outputs из конфигурации ---
    outputs = cfg_st.outputs or {}

    st_tfs = outputs.get("supertrend_lines", []) or []
    cci_cfg = outputs.get("cci_value", {}) or {}
    cci_tfs = cci_cfg.get("tfs") or []

    cci_sign_cfg = outputs.get("cci_sign", {}) or {}
    cci_sign_tfs = cci_sign_cfg.get("tfs") or []

    diffs_cfg = outputs.get("diffs", {}) or {}
    diffs_enabled = diffs_cfg.get("enabled", False)
    diffs_tfs = diffs_cfg.get("tfs") or []
    diffs_ohlc = diffs_cfg.get("ohlc") or ["open", "high", "low", "close"]

    # какие TF вообще нужны для вычислений
    tfs_cfg = cfg_st.tfs or [working_tf]
    active_tfs = _sort_tfs_by_tf_order(
        list(st_tfs) + list(cci_tfs) + list(cci_sign_tfs) + list(diffs_tfs)
    )
    if not active_tfs:
        # ничего не запрошено в outputs — можно ничего не считать
        return

    # на всякий случай, если рабочий TF не указан явно, но используется где-то:
    if working_tf in tfs_cfg and working_tf not in active_tfs:
        active_tfs.append(working_tf)

    # --- 1) рабочий TF (без ресемплинга) ---
    if working_tf in active_tfs:
        st_series, dir_series, cci_series = compute_supertrend_and_cci(
            df, atr_period, multiplier, cci_period
        )

        if working_tf in st_tfs:
            df[f"st_{working_tf}"] = st_series
            df[f"st_dir_{working_tf}"] = dir_series

        if cci_cfg.get("enabled", False) and working_tf in cci_tfs:
            df[f"cci_{working_tf}"] = cci_series

        if cci_sign_cfg.get("enabled", False) and working_tf in cci_sign_tfs:
            df[f"cci_sign_{working_tf}"] = np.sign(cci_series)

        if diffs_enabled and working_tf in diffs_tfs and working_tf in st_tfs:
            for col in diffs_ohlc:
                if col in df.columns:
                    df[f"{col}_minus_st_{working_tf}"] = (
                        df[col].astype(float) - st_series
                    )

    # --- 2) старшие TF (H4, D1 и т.п.) через ресемплинг ---
    # берём только те TF, которые:
    #   а) присутствуют в cfg.supertrend.tfs,
    #   б) реально нужны по outputs,
    #   в) отличаются от working_tf.
    higher_tfs = [
        tf for tf in active_tfs
        if tf != working_tf and tf in (tfs_cfg or [])
    ]

    print("DEBUG MTF: active_tfs =", active_tfs)
    print("DEBUG MTF: tfs_cfg    =", tfs_cfg)
    print("DEBUG MTF: working_tf =", working_tf)
    print("DEBUG MTF: higher_tfs =", higher_tfs)

    if not higher_tfs:
        return

    # базовый df для ресемплинга
    base_ohlc = df[["time", "open", "high", "low", "close"]].copy()

    for tf in higher_tfs:
        # 2.1. ресемплинг OHLC на старший TF
        try:
            df_htf = resample_ohlc(base_ohlc, tf)
        except ValueError as e:
            print(f"[feature_engine] WARNING: cannot resample to {tf}: {e}")
            continue

        if df_htf.empty:
            continue

        # 2.2. считаем SuperTrend/CCI на старшем TF
        st_htf, dir_htf, cci_htf = compute_supertrend_and_cci(
            df_htf, atr_period, multiplier, cci_period
        )

        # 2.3. готовим таблицу фич старшего TF для merge_asof
        cols = []
        df_feat_htf = df_htf[["time"]].copy()

        if tf in st_tfs:
            col_st = f"st_{tf}"
            col_dir = f"st_dir_{tf}"
            df_feat_htf[col_st] = st_htf
            df_feat_htf[col_dir] = dir_htf
            cols.extend([col_st, col_dir])

        if cci_cfg.get("enabled", False) and tf in cci_tfs:
            col_cci = f"cci_{tf}"
            df_feat_htf[col_cci] = cci_htf
            cols.append(col_cci)

        if cci_sign_cfg.get("enabled", False) and tf in cci_sign_tfs:
            col_cci_sign = f"cci_sign_{tf}"
            df_feat_htf[col_cci_sign] = np.sign(cci_htf)
            cols.append(col_cci_sign)

        if not cols:
            # для этого TF ничего не нужно
            continue

        # 2.4. подтягиваем фичи со старшего TF к рабочему через merge_asof
        merged = align_higher_tf_to_working(df, df_feat_htf, cols)

        # добавляем НОВЫЕ колонки в текущий df
        for col in cols:
            df[col] = merged[col]

        # 2.5. diffs для старшего TF (после того как st_<tf> уже добавлен)
        if diffs_enabled and tf in diffs_tfs and tf in st_tfs:
            st_col = f"st_{tf}"
            if st_col in df.columns:
                st_series_aligned = df[st_col].astype(float)
                for col in diffs_ohlc:
                    if col in df.columns:
                        df[f"{col}_minus_st_{tf}"] = (
                                df[col].astype(float) - st_series_aligned
                        )
@register_feature(
    "murrey",
    default_enabled=False,
    lookback_fn=murrey_lookback,
)
def feature_murrey(df: pd.DataFrame, ctx: FeatureContext) -> None:
    profile = ctx.profile
    engine = ctx.engine
    cfg_m = ctx.cfg.murrey

    if not getattr(profile, "use_murrey", True):
        return
    if not cfg_m.enabled:
        return

    required_cols = {"time", "open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        print("[feature_engine] WARNING: cannot compute Murrey – OHLC/time missing")
        return

    try:
        working_tf = SETTINGS.market.working_timeframe
    except Exception:
        working_tf = "H1"

    period_bars = cfg_m.period_bars
    include_extremes = getattr(cfg_m, "include_extremes", True)

    outputs = cfg_m.outputs or {}
    out_levels = bool(outputs.get("levels", True))
    out_dist = bool(outputs.get("distances", True))
    out_zone = bool(outputs.get("zone", True))
    out_pos = bool(outputs.get("pos_in_zone", True))

    tfs_cfg = cfg_m.tfs or [working_tf]
    # хотим от младшего к старшему
    active_tfs = _sort_tfs_by_tf_order(tfs_cfg)

    base_ohlc = df[["time", "open", "high", "low", "close"]].copy()

    def _emit(tf: str, target_df: pd.DataFrame, mur: dict[str, pd.Series]) -> None:
        # уровни
        if out_levels:
            # уровни 0..8 всегда
            for i in range(0, 9):
                target_df[f"mur_{i}_8_{tf}"] = mur[f"mur_{i}_8"]
            # экстремы -2,-1,9,10 если включено
            if include_extremes:
                for i in (-2, -1, 9, 10):
                    key = f"mur_{i}_8"
                    if key in mur:
                        target_df[f"mur_{i}_8_{tf}"] = mur[key]

        # зона/позиция
        if out_zone:
            target_df[f"mur_zone_{tf}"] = mur["mur_zone"]
        if out_pos:
            target_df[f"mur_pos_in_zone_{tf}"] = mur["mur_pos_in_zone"]

        # distances
        if out_dist:
            target_df[f"mur_nearest_idx_{tf}"] = mur["mur_nearest_idx"]
            target_df[f"mur_dist_close_to_nearest_{tf}"] = mur["mur_dist_close_to_nearest"]
            target_df[f"mur_dist_close_to_0_8_{tf}"] = mur["mur_dist_close_to_0_8"]
            target_df[f"mur_dist_close_to_4_8_{tf}"] = mur["mur_dist_close_to_4_8"]
            target_df[f"mur_dist_close_to_8_8_{tf}"] = mur["mur_dist_close_to_8_8"]

    # 1) working TF (без ресемплинга)
    if working_tf in active_tfs:
        mur = compute_murrey_grid(df, period_bars=period_bars, include_extremes=include_extremes)
        _emit(working_tf, df, mur)

    # 2) higher TF через ресемплинг + merge_asof
    higher_tfs = [tf for tf in active_tfs if tf != working_tf]
    for tf in higher_tfs:
        try:
            df_htf = resample_ohlc(base_ohlc, tf)
        except ValueError as e:
            print(f"[feature_engine] WARNING: cannot resample to {tf}: {e}")
            continue
        if df_htf.empty:
            continue

        mur_htf = compute_murrey_grid(df_htf, period_bars=period_bars, include_extremes=include_extremes)

        # формируем df_feat_htf только с нужными колонками
        df_feat_htf = df_htf[["time"]].copy()
        cols = []

        # временно “emit” в df_feat_htf, затем merge и подставим в df
        _emit(tf, df_feat_htf, mur_htf)
        cols = [c for c in df_feat_htf.columns if c != "time"]

        if not cols:
            continue

        merged = align_higher_tf_to_working(df, df_feat_htf, cols)
        for c in cols:
            df[c] = merged[c]



