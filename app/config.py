import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "./data/media")).resolve()
FRAME_DIR = Path(os.getenv("FRAME_DIR", "./data/frames")).resolve()
RETRIEVED_DIR = Path(os.getenv("RETRIEVED_DIR", "./data/retrieved")).resolve()
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./data/cache")).resolve()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen/qwen3-vl-8b-instruct")
TEXT_MODEL = os.getenv("TEXT_MODEL") or VLM_MODEL
VLM_STRONG_MODEL = os.getenv("VLM_STRONG_MODEL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FRAME_EMBEDDING_MODEL = os.getenv("FRAME_EMBEDDING_MODEL", "clip-ViT-B-32")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "15"))


def ensure_directories() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
