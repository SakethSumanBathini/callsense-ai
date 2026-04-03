"""
CallSense AI - Unified LLM Service (v2.0)
Gemini structured JSON output + httpx connection pooling + fallback chain.
"""

import json
import re
import logging
import asyncio
from typing import Optional, Dict, Any

import httpx
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from typing import List, Literal

from src.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# HTTPX CONNECTION POOL SINGLETON (saves 200-500ms per request)
# =============================================================================
_http_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    """Get or create singleton httpx client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        try:
            _http_client = httpx.AsyncClient(
                timeout=120.0,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                http2=True,
            )
        except ImportError:
            # h2 not installed, fallback to http1.1
            _http_client = httpx.AsyncClient(
                timeout=120.0,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
    return _http_client


# =============================================================================
# GEMINI STRUCTURED OUTPUT SCHEMA
# =============================================================================
class AnalysisSOPSchema(BaseModel):
    greeting: bool
    identification: bool
    problemStatement: bool
    solutionOffering: bool
    closing: bool
    complianceScore: float
    adherenceStatus: str
    explanation: str

class AnalyticsSchema(BaseModel):
    paymentPreference: str
    rejectionReason: str
    sentiment: str

class AnalysisOutputSchema(BaseModel):
    summary: str
    sop_validation: AnalysisSOPSchema
    analytics: AnalyticsSchema
    keywords: List[str]


# =============================================================================
# PROVIDER CONFIGS
# =============================================================================
LLM_PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "key_attr": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "llama-3.3-70b-versatile",
        "key_attr": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "key_attr": "OPENROUTER_API_KEY",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1/",
        "model": "meta/llama-3.1-70b-instruct",
        "key_attr": "NVIDIA_API_KEY",
    },
}
FALLBACK_ORDER = ["gemini", "groq", "openrouter", "nvidia"]


async def call_llm_openai_compatible(
    prompt: str, provider: str, temperature: float = 0.1, max_tokens: int = 4096,
) -> Optional[str]:
    """Call LLM using OpenAI-compatible format with connection pooling."""
    config = LLM_PROVIDERS.get(provider)
    if not config:
        return None

    api_key = getattr(settings, config["key_attr"], "")
    if not api_key:
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://callsense-ai.com"
        headers["X-Title"] = "CallSense AI"

    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "You are a precise call center analytics AI. Return ONLY valid JSON, no markdown, no backticks."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        client = get_http_client()
        # Retry up to 3 times with exponential backoff for rate limits
        for attempt in range(3):
            response = await client.post(
                f"{config['base_url']}chat/completions",
                headers=headers, json=payload
            )
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.info(f"LLM [{provider}] responded: {len(content)} chars")
                return content
            elif response.status_code == 429:
                # Rate limited — wait and retry with exponential backoff
                wait_time = (2 ** attempt) + 1  # 2s, 3s, 5s
                logger.warning(f"LLM [{provider}] rate limited (429), retry in {wait_time}s (attempt {attempt+1}/3)")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.warning(f"LLM [{provider}] error {response.status_code}: {response.text[:200]}")
                return None
        logger.warning(f"LLM [{provider}] exhausted retries after rate limiting")
        return None
    except Exception as e:
        logger.error(f"LLM [{provider}] exception: {e}")
        return None


async def call_gemini_native(prompt: str) -> Optional[str]:
    """Call Gemini using native SDK with structured JSON output."""
    if not settings.GEMINI_API_KEY:
        return None

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Try structured output first (guaranteed valid JSON)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.05,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=AnalysisOutputSchema,
                ),
            )
            text = response.text
            logger.info(f"Gemini structured output: {len(text)} chars")
            return text
        except Exception as e:
            logger.warning(f"Gemini structured output failed: {e}, falling back to free-form")
            # Fallback to regular generation
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            text = response.text
            logger.info(f"Gemini free-form responded: {len(text)} chars")
            return text

    except Exception as e:
        logger.error(f"Gemini native error: {e}")
        return None


async def call_llm_with_fallback(
    prompt: str, preferred_provider: Optional[str] = None
) -> Optional[str]:
    """Call LLM with automatic fallback chain."""
    provider = preferred_provider or settings.LLM_PROVIDER
    order = [provider] + [p for p in FALLBACK_ORDER if p != provider]

    for p in order:
        logger.info(f"Trying LLM provider: {p}")
        if p == "gemini":
            result = await call_gemini_native(prompt)
            if result:
                return result
            result = await call_llm_openai_compatible(prompt, p)
            if result:
                return result
        else:
            result = await call_llm_openai_compatible(prompt, p)
            if result:
                return result
        logger.warning(f"Provider {p} failed, trying next...")
        await asyncio.sleep(0.3)

    logger.error("ALL LLM providers failed!")
    return None


def parse_llm_json(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling all edge cases."""
    if not response_text:
        return None

    text = response_text.strip()

    # Remove markdown fences
    fence_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON object with brace matching
    try:
        start = text.index("{")
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0: end = i + 1; break
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # Fix Python booleans
    try:
        fixed = re.sub(r'\bTrue\b', 'true', text)
        fixed = re.sub(r'\bFalse\b', 'false', fixed)
        fixed = re.sub(r'\bNone\b', 'null', fixed)
        start = fixed.index("{")
        depth = 0
        end = start
        for i in range(start, len(fixed)):
            if fixed[i] == "{": depth += 1
            elif fixed[i] == "}":
                depth -= 1
                if depth == 0: end = i + 1; break
        return json.loads(fixed[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse LLM JSON: {e}")
        return None