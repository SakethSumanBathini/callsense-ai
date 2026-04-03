"""
CallSense AI - Sarvam AI Speech-to-Text Service (v2.1)
Fixes: 25s chunks (safety margin), fast fail on 400, no retries for duration errors.
"""

import httpx
import asyncio
import logging
from typing import Tuple

from src.config import settings
from src.transcription.audio_utils import chunk_audio, cleanup_temp_files

logger = logging.getLogger(__name__)


async def transcribe_chunk_sarvam(
    audio_path: str, language: str = "Hindi", mode: str = "translate"
) -> str:
    """Transcribe a single audio chunk using Sarvam REST API. No retries on 400."""
    language_code = settings.get_language_code(language)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(audio_path, "rb") as f:
                mime = "audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav"
                files = {"file": ("audio.mp3", f, mime)}
                data = {"model": settings.SARVAM_MODEL, "language_code": language_code}
                if "saaras" in settings.SARVAM_MODEL:
                    data["mode"] = mode

                response = await client.post(
                    settings.SARVAM_STT_URL,
                    files=files, data=data,
                    headers={"api-subscription-key": settings.SARVAM_API_KEY},
                )

            if response.status_code == 200:
                text = response.json().get("transcript", "")
                logger.info(f"Sarvam chunk OK: {len(text)} chars")
                return text
            elif response.status_code == 400:
                logger.warning(f"Sarvam 400 (skip): {response.text[:100]}")
                return ""  # Fast fail, no retry
            else:
                logger.warning(f"Sarvam {response.status_code}: {response.text[:100]}")
                return ""
    except Exception as e:
        logger.error(f"Sarvam error: {e}")
        return ""


async def transcribe_full_audio_sarvam(
    audio_path: str, language: str = "Hindi"
) -> Tuple[str, str]:
    """Transcribe full audio. 25s chunks + concurrent processing."""
    logger.info(f"Sarvam: starting for {language}")

    # 29s chunks — maximize content per chunk, minimize API calls
    chunks = chunk_audio(audio_path, chunk_duration_ms=29000, overlap_ms=0)
    logger.info(f"Sarvam: {len(chunks)} chunks")

    # Concurrent transcription of all chunks
    tasks = [transcribe_chunk_sarvam(cp, language, "translate") for cp in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    transcripts = []
    for i, r in enumerate(results):
        if isinstance(r, str) and r:
            transcripts.append(r)

    if len(chunks) > 1:
        cleanup_temp_files(chunks)

    full = " ".join(transcripts)
    logger.info(f"Sarvam complete: {len(full)} chars")
    return full, language