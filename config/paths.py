from pathlib import Path

# Базовая директория проекта (где лежит RL_PyTorch)
BASE_DIR = Path(__file__).resolve().parents[1]

# Каталоги данных
DATA_DIR = BASE_DIR / "data"
MASTER_DIR = DATA_DIR / "master"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
LIVE_LOG_DIR = DATA_DIR / "live_logs"

# Создаём каталоги, если их нет
for d in (DATA_DIR, MASTER_DIR, SNAPSHOT_DIR, LIVE_LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
