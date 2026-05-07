"""
config.py — central settings loader
reads from .env file and environment variables
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # HuggingFace
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Models
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "openai/clip-vit-large-patch14")
    CLIP_MODEL_2: str = os.getenv("CLIP_MODEL_2", "openai/clip-vit-base-patch32")
    USE_ENSEMBLE: bool = os.getenv("USE_ENSEMBLE", "true").lower() == "true"

    # Detection
    DETECTION_THRESHOLD: float = float(os.getenv("DETECTION_THRESHOLD", "0.9"))
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.16"))
    MIN_AI_MARGIN: float = float(os.getenv("MIN_AI_MARGIN", "0.08"))
    USE_ROBUST_AUGMENTATION: bool = os.getenv("USE_ROBUST_AUGMENTATION", "true").lower() == "true"

    # App
    APP_TITLE: str = os.getenv("APP_TITLE", "AI Image Detector")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    MODEL_CACHE_DIR: Path = BASE_DIR / os.getenv("MODEL_CACHE_DIR", "models/cache")

    def __init__(self):
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Tell HuggingFace where to cache models
        os.environ["TRANSFORMERS_CACHE"] = str(self.MODEL_CACHE_DIR)
        os.environ["HF_HOME"] = str(self.MODEL_CACHE_DIR)
        if self.HF_TOKEN:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.HF_TOKEN


settings = Settings()
