from pathlib import Path
from config.settings import SETTINGS

# Базовая директория проекта (где лежит RL_PyTorch)
BASE_DIR = Path(__file__).resolve().parents[1]

# --- roots (по умолчанию, если YAML не задан) ---
DATA_DIR = BASE_DIR / "data"
DEFAULT_EXTERNAL_ROOT = DATA_DIR / "external"
DEFAULT_MASTER_ROOT = DATA_DIR / "master"
DEFAULT_SNAPSHOTS_ROOT = DATA_DIR / "snapshots"

# live_logs пока оставляем общими (не привязаны к источнику)
LIVE_LOG_DIR = DATA_DIR / "live_logs"

def _active_subdir(kind: str, default_name: str) -> str:
    """
    kind: external_dir | master_dir | snapshots_dir
    """
    src = getattr(SETTINGS, "data_source", "dukascopy") or "dukascopy"
    sources = getattr(SETTINGS, "sources", {}) or {}
    s_cfg = sources.get(src, {}) or {}
    return str(s_cfg.get(kind, default_name))

def _active_root(kind: str, default_root: Path) -> Path:
    """
    kind: external_root | master_root | snapshots_root
    """
    layout = getattr(SETTINGS, "data_layout", {}) or {}
    rel = layout.get(kind)
    if rel:
        return BASE_DIR / rel
    return default_root

# --- ACTIVE dirs (по data_source) ---
EXTERNAL_ROOT = _active_root("external_root", DEFAULT_EXTERNAL_ROOT)
MASTER_ROOT = _active_root("master_root", DEFAULT_MASTER_ROOT)
SNAPSHOTS_ROOT = _active_root("snapshots_root", DEFAULT_SNAPSHOTS_ROOT)

EXTERNAL_DIR = EXTERNAL_ROOT / _active_subdir("external_dir", "dukascopy")
MASTER_DIR = MASTER_ROOT / _active_subdir("master_dir", "dukascopy")
SNAPSHOT_DIR = SNAPSHOTS_ROOT / _active_subdir("snapshots_dir", "dukascopy")

# Backward-compatible alias (в других модулях используется SNAPSHOT_DIR)
SNAPSHOT_DIR = SNAPSHOT_DIR

# Создаём каталоги, если их нет
for d in (DATA_DIR, EXTERNAL_ROOT, MASTER_ROOT, SNAPSHOTS_ROOT, EXTERNAL_DIR, MASTER_DIR, SNAPSHOT_DIR, LIVE_LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
