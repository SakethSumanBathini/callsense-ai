"""
CallSense AI - Audio Utilities
Handles base64 decoding, MP3 processing, audio chunking, and preprocessing.
Includes edge case handling for corrupted audio, silent audio, large files.
"""

import base64
import io
import os
import tempfile
import logging
import hashlib
from typing import List, Tuple
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Maximum audio file size (25MB)
MAX_AUDIO_SIZE = 25 * 1024 * 1024


def decode_base64_audio(audio_base64: str) -> bytes:
    """
    Decode base64-encoded audio to raw bytes.
    Handles standard base64, URL-safe base64, and data URL prefixes.
    """
    try:
        cleaned = audio_base64.strip()

        # Remove data URL prefix if present
        if "," in cleaned and cleaned.startswith("data:"):
            cleaned = cleaned.split(",", 1)[1]

        # Remove whitespace
        cleaned = cleaned.replace("\n", "").replace("\r", "").replace(" ", "")

        # Add padding if needed
        padding_needed = len(cleaned) % 4
        if padding_needed:
            cleaned += "=" * (4 - padding_needed)

        # Try standard base64 first
        try:
            audio_bytes = base64.b64decode(cleaned)
        except Exception:
            audio_bytes = base64.urlsafe_b64decode(cleaned)

        if len(audio_bytes) < 4:
            raise ValueError("Decoded audio is too small - likely invalid base64")

        # Check file size limit
        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise ValueError(f"Audio file too large: {len(audio_bytes)} bytes (max {MAX_AUDIO_SIZE})")

        logger.info(f"Successfully decoded base64 audio: {len(audio_bytes)} bytes")
        return audio_bytes

    except Exception as e:
        logger.error(f"Base64 decoding failed: {e}")
        raise ValueError(f"Failed to decode base64 audio: {str(e)}")


def compute_audio_hash(audio_bytes: bytes) -> str:
    """Compute SHA-256 hash of audio for caching."""
    return hashlib.sha256(audio_bytes).hexdigest()[:16]


def save_audio_to_temp(audio_bytes: bytes, format: str = "mp3") -> str:
    """Save audio bytes to a temporary file."""
    temp_dir = tempfile.mkdtemp(prefix="callsense_")
    temp_path = os.path.join(temp_dir, f"audio.{format}")

    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    logger.info(f"Saved audio to temp file: {temp_path}")
    return temp_path


def preprocess_audio(audio_path: str) -> str:
    """
    Preprocess audio for optimal Sarvam performance:
    - Convert to 16kHz mono WAV (Sarvam optimal format)
    - Normalize volume
    - Returns path to preprocessed file
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Normalize volume
        target_dBFS = -20
        change_in_dBFS = target_dBFS - audio.dBFS
        if abs(change_in_dBFS) > 1:  # Only normalize if significantly off
            audio = audio.apply_gain(change_in_dBFS)
        
        # Save as WAV (Sarvam processes WAV faster)
        temp_dir = tempfile.mkdtemp(prefix="callsense_pre_")
        preprocessed_path = os.path.join(temp_dir, "preprocessed.wav")
        audio.export(preprocessed_path, format="wav")
        
        original_size = os.path.getsize(audio_path)
        new_size = os.path.getsize(preprocessed_path)
        logger.info(
            f"Audio preprocessed: {original_size} → {new_size} bytes "
            f"({int((1 - new_size/max(original_size,1))*100)}% size change), 16kHz mono WAV"
        )
        
        return preprocessed_path
    except Exception as e:
        logger.warning(f"Audio preprocessing failed, using original: {e}")
        return audio_path


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
        logger.info(f"Audio duration: {duration:.1f} seconds")
        return duration
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        return 0.0


def is_silent_audio(audio_path: str, silence_threshold: float = -45.0) -> bool:
    """Check if audio is essentially silent (no speech)."""
    try:
        audio = AudioSegment.from_file(audio_path)
        return audio.dBFS < silence_threshold
    except Exception:
        return False


def chunk_audio(
    audio_path: str,
    chunk_duration_ms: int = 29000,
    overlap_ms: int = 0
) -> List[str]:
    """
    Split audio file into chunks for Sarvam REST API (30s limit).
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        total_duration = len(audio)

        logger.info(
            f"Chunking audio: {total_duration/1000:.1f}s total, "
            f"{chunk_duration_ms/1000}s per chunk"
        )

        if total_duration <= chunk_duration_ms + 1000:
            logger.info("Audio short enough, no chunking needed")
            return [audio_path]

        chunks = []
        temp_dir = tempfile.mkdtemp(prefix="callsense_chunks_")
        start = 0
        chunk_idx = 0

        while start < total_duration:
            end = min(start + chunk_duration_ms, total_duration)
            chunk = audio[start:end]

            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx:03d}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunks.append(chunk_path)

            start = end - overlap_ms if end < total_duration else end
            chunk_idx += 1

        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks

    except Exception as e:
        logger.error(f"Audio chunking failed: {e}")
        return [audio_path]


def cleanup_temp_files(paths: List[str]) -> None:
    """Remove temporary audio files and their directories."""
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
            parent = os.path.dirname(path)
            if parent and os.path.exists(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except Exception:
            pass
