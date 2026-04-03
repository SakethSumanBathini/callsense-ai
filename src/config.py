"""
CallSense AI - Configuration Module
Centralized settings with environment variable validation.
"""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # API Authentication
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "sk_track3_987654321")

    # Sarvam AI (Primary ASR)
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
    SARVAM_STT_TRANSLATE_URL: str = "https://api.sarvam.ai/speech-to-text-translate"
    SARVAM_BATCH_URL: str = "https://api.sarvam.ai/speech-to-text/batch"

    # Google Gemini (Primary LLM)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Groq (Fallback LLM + ASR)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # OpenRouter (Secondary Fallback)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # NVIDIA NIMs (Tertiary Fallback)
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")

    # LLM Provider Selection
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Audio Processing
    MAX_CHUNK_DURATION_SECONDS: int = 29  # Sarvam REST API limit is 30s
    SUPPORTED_AUDIO_FORMATS: list = ["mp3", "wav", "m4a", "aac", "ogg", "flac"]

    # Sarvam STT Model Config
    SARVAM_MODEL: str = "saaras:v3"

    # Language mapping for Sarvam
    LANGUAGE_MAP: dict = {
        "Tamil": "ta-IN",
        "Hindi": "hi-IN",
        "English": "en-IN",
        "Bengali": "bn-IN",
        "Telugu": "te-IN",
        "Kannada": "kn-IN",
        "Malayalam": "ml-IN",
        "Marathi": "mr-IN",
        "Gujarati": "gu-IN",
        "Punjabi": "pa-IN",
        "Odia": "od-IN",
    }

    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 1.0

    # Vector Store
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    @classmethod
    def get_language_code(cls, language: str) -> str:
        """Convert language name to BCP-47 code for Sarvam API."""
        return cls.LANGUAGE_MAP.get(language, "hi-IN")

    @classmethod
    def validate(cls) -> list:
        """Validate that required API keys are set. Returns list of warnings."""
        warnings = []
        if not cls.SARVAM_API_KEY:
            warnings.append("SARVAM_API_KEY not set - primary ASR will not work")
        if not cls.GEMINI_API_KEY:
            warnings.append("GEMINI_API_KEY not set - primary LLM will not work")
        if not cls.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY not set - fallback will not work")
        return warnings


settings = Settings()
