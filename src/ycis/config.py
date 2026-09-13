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
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", 1000))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", 1000))

    # 3. 모델 설정
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral-small:latest")

config = Config()
