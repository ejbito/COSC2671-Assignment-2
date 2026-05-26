from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SENTIMENT_METHOD = os.getenv("SENTIMENT_METHOD", "vader").strip().lower()
if SENTIMENT_METHOD not in {"vader", "roberta"}:
    raise ValueError("SENTIMENT_METHOD must be either 'vader' or 'roberta'.")

RUN_DIR = DATA_DIR / SENTIMENT_METHOD
PROCESSED_DIR = RUN_DIR / "processed"
OUTPUT_DIR = RUN_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
GRAPHS_DIR = OUTPUT_DIR / "graphs"


PROJECT_SEKAI_LABEL = "Project Sekai"
BANDORI_LABEL = "BanG Dream"

PROJECT_SEKAI_RAW = RAW_DIR / "project_sekai_raw.json"
BANDORI_RAW = RAW_DIR / "bandori_raw.json"

PROJECT_SEKAI_CLEAN = PROCESSED_DIR / "project_sekai_comments_clean.csv"
BANDORI_CLEAN = PROCESSED_DIR / "bandori_comments_clean.csv"
COMBINED_CLEAN = PROCESSED_DIR / "combined_comments_clean.csv"
COMBINED_WITH_NETWORK = PROCESSED_DIR / "combined_comments_with_network.csv"

PROJECT_SEKAI_EDGES = PROCESSED_DIR / "project_sekai_edges.csv"
BANDORI_EDGES = PROCESSED_DIR / "bandori_edges.csv"
COMBINED_EDGES = PROCESSED_DIR / "combined_edges.csv"


PROJECT_SEKAI_QUERIES = [
    "Project Sekai update",
    "Project Sekai tier list",
    "Project Sekai opinions",
    "Project Sekai review",
    "Project Sekai community",
]

BANDORI_QUERIES = [
    "Bandori",
    "Girls Band Party",
    "Bandori gameplay",
    "Bandori tier list",
    "Girls Band Party update",
]


def ensure_dirs():
    for path in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, GRAPHS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
