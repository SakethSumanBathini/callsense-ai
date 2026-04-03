"""
CallSense AI - Whisper Fallback Service (v2.1)
Uses Groq Whisper Large V3 TRANSLATION mode for direct English output.
"""

import logging
import os
from typing import Tuple
from groq import Groq

from src.config import settings

logger = logging.getLogger(__name__)


async def transcribe_with_whisper(
    audio_path: str,
    language: str = "Hindi"
) -> Tuple[str, str]:
    """
    Transcribe audio using Groq Whisper Large V3.
    Uses TRANSLATION mode → outputs ENGLISH from any language.
    Falls back to transcription mode if translation fails.
    """
    if not settings.GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set")
        return "", language

    try:
        client = Groq(api_key=settings.GROQ_API_KEY)

        # Try TRANSLATION mode with retry for rate limits
        for attempt in range(3):
            try:
                with open(audio_path, "rb") as audio_file:
                    result = client.audio.translations.create(
                        file=("audio.mp3", audio_file),
                        model="whisper-large-v3",
                        response_format="text",
                        temperature=0.0,
                    )

                transcript = str(result).strip()
                if transcript and len(transcript) > 50:
                    logger.info(f"Whisper translation OK: {len(transcript)} chars (English)")
                    return transcript, language
                else:
                    logger.warning(f"Whisper translation too short: {len(transcript)} chars")
                    break  # Don't retry if result is just short
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    import asyncio
                    wait = (2 ** attempt) + 1
                    logger.warning(f"Whisper rate limited, retry in {wait}s (attempt {attempt+1}/3)")
                    await asyncio.sleep(wait)
                    continue
                logger.warning(f"Whisper translation failed: {e}")
                break

        # Fallback: transcription mode (original language)
        logger.info("Trying Whisper transcription mode...")
        whisper_lang_map = {
            "Hindi": "hi", "Tamil": "ta", "English": "en",
            "Bengali": "bn", "Telugu": "te", "Kannada": "kn",
            "Malayalam": "ml", "Marathi": "mr",
        }

        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=("audio.mp3", audio_file),
                model="whisper-large-v3",
                language=whisper_lang_map.get(language, "hi"),
                response_format="text",
                temperature=0.0,
            )

        transcript = str(result).strip()
        logger.info(f"Whisper transcription OK: {len(transcript)} chars")
        return transcript, language

    except Exception as e:
        logger.error(f"Whisper failed: {e}")
        return "", language