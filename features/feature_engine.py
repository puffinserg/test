# features/feature_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Callable, Dict, Optional, List, Any

import numpy as np
import pandas as pd

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
    prev_close = df["close"].shift(1)

    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - prev_close).abs()
    low_prev = (df["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=period, min_periods=1).mean()


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

    for p in shorts:
        col = f"ret_{p}"
        df[col] = close.pct_change(periods=p).fillna(0.0)

    col_long = f"ret_long_{long_p}"
    df[col_long] = close.pct_change(periods=long_p).fillna(0.0)


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
    returns_1 = df["close"].pct_change().fillna(0.0)
    df["volatility"] = (
        returns_1.rolling(window=period, min_periods=1).std().fillna(0.0)
    )


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

    df["spread_mean"] = (
        spread.rolling(window=period, min_periods=1).mean().fillna(0.0)
    )
    df["spread_std"] = (
        spread.rolling(window=period, min_periods=1).std().fillna(0.0)
    )

    if "atr" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = df["spread_mean"] / df["atr"]
            ratio = ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["spread_over_atr"] = ratio
    else:
        df["spread_over_atr"] = 0.0

@register_feature(
    "supertrend",
    default_enabled=False,
    lookback_fn=supertrend_lookback,
)
def feature_supertrend(df: pd.DataFrame, ctx: FeatureContext) -> None:
    """
    Реализация SuperTrend + CCI для рабочего таймфрейма.

    Что делаем:
    - считаем ATR c периодом supertrend_atr_period (из профиля или из cfg.supertrend);
    - строим классический SuperTrend (верхняя/нижняя ленты + линия тренда);
    - считаем CCI по typical price;
    - в зависимости от settings.yaml создаём колонки:
        st_<TF>, cci_<TF>, cci_sign_<TF>, <OHLC>_minus_st_<TF>.
    Пока считаем только для рабочего ТФ (market.working_timeframe), MTF сделаем позже.
    """
    profile = ctx.profile
    engine = ctx.engine
    cfg_st = ctx.cfg.supertrend

    # отключено профилем или глобально
    if not getattr(profile, "use_supertrend", True):
        return
    if not cfg_st.enabled:
        return

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df.columns):
        print("[feature_engine] WARNING: cannot compute SuperTrend – OHLC missing")
        return

    # Рабочий таймфрейм (например, "H1")
    try:
        from config.settings import SETTINGS  # уже импортирован выше, но страховка на случай refactor
        working_tf = SETTINGS.market.working_timeframe
    except Exception:
        working_tf = "H1"

    # --- Параметры из профиля / конфигурации ---
    atr_period = engine._eff_st_atr_period()
    multiplier = engine._eff_st_multiplier()
    cci_period = engine._eff_st_cci_period()

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # --- ATR для SuperTrend (локальный, независимый от "atr") ---
    prev_close = close.shift(1)
    high_low = high - low
    high_prev = (high - prev_close).abs()
    low_prev = (low - prev_close).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    atr_st = tr.rolling(window=atr_period, min_periods=1).mean()

    # --- Базовые линии upper / lower ---
    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr_st
    basic_lb = hl2 - multiplier * atr_st

    ub = basic_ub.to_numpy().copy()
    lb = basic_lb.to_numpy().copy()
    c = close.to_numpy()
    n = len(df)

    # финальные ленты по классическому алгоритму
    for i in range(1, n):
        # верхняя лента
        if (basic_ub.iat[i] < ub[i - 1]) or (c[i - 1] > ub[i - 1]):
            ub[i] = basic_ub.iat[i]
        else:
            ub[i] = ub[i - 1]

        # нижняя лента
        if (basic_lb.iat[i] > lb[i - 1]) or (c[i - 1] < lb[i - 1]):
            lb[i] = basic_lb.iat[i]
        else:
            lb[i] = lb[i - 1]

    st = np.full(n, np.nan, dtype=float)
    direction = np.zeros(n, dtype=int)  # +1 = up, -1 = down

    # инициализация
    st[0] = ub[0]
    direction[0] = -1 if c[0] < ub[0] else 1

    for i in range(1, n):
        prev_st = st[i - 1]
        prev_ub = ub[i - 1]
        prev_lb = lb[i - 1]

        # предыдущий тренд вниз (линия = верхняя лента)
        if prev_st == prev_ub:
            if c[i] <= ub[i]:
                st[i] = ub[i]
                direction[i] = -1
            else:
                st[i] = lb[i]
                direction[i] = 1
        # предыдущий тренд вверх (линия = нижняя лента)
        elif prev_st == prev_lb:
            if c[i] >= lb[i]:
                st[i] = lb[i]
                direction[i] = 1
            else:
                st[i] = ub[i]
                direction[i] = -1
        else:
            # на всякий случай fallback
            st[i] = st[i - 1]
            direction[i] = direction[i - 1]

    st_series = pd.Series(st, index=df.index)

    # --- CCI по typical price ---
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=cci_period, min_periods=1).mean()
    mean_dev = (tp - sma_tp).abs().rolling(window=cci_period, min_periods=1).mean()
    denom = 0.015 * mean_dev.replace(0, np.nan)
    cci = ((tp - sma_tp) / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # --- Чтение outputs из конфигурации ---
    outputs = cfg_st.outputs or {}

    # Линии SuperTrend
    st_tfs = outputs.get("supertrend_lines", [])
    if working_tf in st_tfs:
        df[f"st_{working_tf}"] = st_series
        # направление тренда (по желанию, YAML об этом не знает, но это полезная фича)
        df[f"st_dir_{working_tf}"] = direction

    # CCI value
    cci_cfg = outputs.get("cci_value", {}) or {}
    if cci_cfg.get("enabled", False) and working_tf in (cci_cfg.get("tfs") or []):
        df[f"cci_{working_tf}"] = cci

    # CCI sign (-1/0/1)
    cci_sign_cfg = outputs.get("cci_sign", {}) or {}
    if cci_sign_cfg.get("enabled", False) and working_tf in (cci_sign_cfg.get("tfs") or []):
        sign = np.sign(cci)
        df[f"cci_sign_{working_tf}"] = sign

    # Разности OHLC - SuperTrend
    diffs_cfg = outputs.get("diffs", {}) or {}
    if diffs_cfg.get("enabled", False) and working_tf in (diffs_cfg.get("tfs") or []):
        ohlc_list = diffs_cfg.get("ohlc") or ["open", "high", "low", "close"]
        for col in ohlc_list:
            if col in df.columns:
                df[f"{col}_minus_st_{working_tf}"] = df[col].astype(float) - st_series


