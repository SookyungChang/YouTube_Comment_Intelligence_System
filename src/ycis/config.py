import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# .env load
load_dotenv()

@dataclass(frozen=True)
class Config:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    JSON_DIR: Path = DATA_DIR / "json"

    # APIs
    YOUTUBE_API_KEY: str = os.getenv('YOUTUBE_API_KEY')
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY')

    # Comment fetching limits
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", 100))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", 3))

    # Inference setting
    TRAINED_MODEL_ID = "sweetguma/bert-sentiment-model"
    BATCH_SIZE: int = 32
    MAX_LENGTH: int = 128 # comment max length

config = Config()
