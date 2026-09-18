import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download

# .env load
load_dotenv()

@dataclass(frozen=True)
class Config:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DB_DIR: Path = DATA_DIR / "DB"
    def __post_init__(self):
        self.DB_DIR.mkdir(parents=True, exist_ok=True)

    # APIs
    YOUTUBE_API_KEY: str = os.getenv('YOUTUBE_API_KEY')
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY')

    # Comment fetching limits
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", 100))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", 3))

    # Inference setting
    TRAINED_MODEL_ID = "sweetguma/bert-sentiment-model"
    TRAINED_MODEL_PATH = snapshot_download(repo_id=TRAINED_MODEL_ID)
    BATCH_SIZE: int = 32
    MAX_LENGTH: int = 128 # comment max length
    SHARED_EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    # LLM setting
    LLM_MODEL_NAME = "openai/gpt-oss-20b"

config = Config()
