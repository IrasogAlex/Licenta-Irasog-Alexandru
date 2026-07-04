from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT_DIR / "models"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
DEFAULT_MODEL_PATH = MODEL_DIR / "xgb_fraud_detector.joblib"
RANDOM_STATE = 42
