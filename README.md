<p align="center">
  <img src="https://img.shields.io/badge/CallSense-AI-00C853?style=for-the-badge&logoColor=white&labelColor=0A0F14" alt="CallSense AI" height="40"/>
</p>

<h1 align="center">CallSense AI</h1>
<h3 align="center">Intelligent Call Center Compliance Analytics for India's Languages</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Track_3-Call_Center_Compliance-4ADE80?style=flat-square" alt="Track 3"/>
  <img src="https://img.shields.io/badge/HCL_GUVI-Hackathon_2026-blue?style=flat-square" alt="HCL GUVI"/>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Sarvam_AI-Saaras_v3-FF6F00?style=flat-square" alt="Sarvam AI"/>
</p>

<p align="center">
  <strong>Processes voice recordings in Hindi (Hinglish) & Tamil (Tanglish) → Transcribes → Validates SOP Compliance → Extracts Business Intelligence</strong>
</p>

---

## Description

Indian call centers handle millions of conversations daily in Hinglish (Hindi-English) and Tanglish (Tamil-English), but no existing compliance tool understands India's code-mixed languages. Manual call review covers less than 2% of calls, leaving 98% unaudited.

**CallSense AI** solves this by:

- Accepting call recordings via API (Base64 MP3)
- Transcribing Hinglish/Tanglish speech with state-of-the-art accuracy
- Validating every call against a 5-step SOP checklist
- Classifying payment preferences and rejection reasons
- Extracting sentiment, keywords, and actionable insights
- Storing transcripts in a vector database for semantic search

**Target Users:** Call center managers, QA teams, compliance officers at Indian BPOs and financial institutions.

**Success = Every call automatically audited, every SOP violation caught, every insight extracted — in seconds, not hours.**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CLIENT REQUEST                            │
│         POST /api/call-analytics (Base64 MP3)                │
│         Header: x-api-key: YOUR_KEY                          │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  API KEY VALIDATION                          │
│              401 Unauthorized if invalid                     │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│              AUDIO PROCESSING PIPELINE                       │
│  Base64 Decode → MP3 Parse (pydub) → Chunk 29s (FFmpeg)     │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│           SPEECH-TO-TEXT (ASR) ENGINE                        │
│                                                              │
│  PRIMARY: Sarvam AI Saaras v3                                │
│  - 19.3% WER on IndicVoices benchmark                       │
│  - Beats GPT-4o Transcribe, Gemini 3 Pro, Deepgram Nova3   │
│  - Native Hinglish/Tanglish code-switching                  │
│  - mode="translate" for English transcript output           │
│                                                              │
│  FALLBACK: Groq Whisper Large v3 (auto-activates on fail)  │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│              AI ANALYSIS ENGINE                              │
│                                                              │
│  Single master prompt extracts ALL fields in one LLM call:  │
│  → Summary → SOP Validation (5 steps + score + status)      │
│  → Payment Preference → Rejection Reason → Sentiment        │
│  → Keywords                                                  │
│                                                              │
│  FALLBACK CHAIN:                                             │
│  Gemini 2.5 Flash → Groq Llama 3.3 → OpenRouter → NVIDIA   │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│            POST-LLM VALIDATION LAYER                        │
│  Pydantic Literal types enforce exact enum values            │
│  Normalizers auto-correct LLM output to valid enums         │
│  Keywords verified against transcript text                   │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│          VECTOR STORAGE (ChromaDB)                           │
│  Transcript chunked → embedded → indexed for search         │
│  Searchable via /api/search endpoint                        │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
              STRUCTURED JSON RESPONSE
```

---

## Tech Stack

| Layer | Technology | Why This Choice |
|:------|:-----------|:---------------|
| **Framework** | FastAPI (Python 3.12) | Async, auto-docs, Pydantic native, fastest Python framework |
| **Primary ASR** | Sarvam AI Saaras v3 | #1 for Indian languages — 19.3% WER, native Hinglish/Tanglish |
| **Fallback ASR** | Groq Whisper Large v3 | 216x real-time speed, automatic failover |
| **Primary LLM** | Google Gemini 2.5 Flash | Best free-tier reasoning, 1M context window |
| **LLM Fallbacks** | Groq Llama 3.3 70B, OpenRouter, NVIDIA NIMs | Zero-downtime guarantee across 4 providers |
| **Vector DB** | ChromaDB | Lightweight, Python-native, semantic search ready |
| **Audio** | pydub + FFmpeg | Robust base64 decode, format handling, chunking |
| **Validation** | Pydantic v2 Literal types | Schema enforcement — invalid responses impossible |
| **Async Tasks** | Celery + Redis | Async voice processing support |
| **Deployment** | Docker + Railway/Render | One-command containerized deployment |
| **Frontend** | HTML/CSS/JS + Chart.js + Tailwind | Zero-dependency dashboard |

---

## Project Structure

```
callsense-ai/
├── README.md                           # Documentation
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore rules
├── Dockerfile                          # Container deployment
│
├── src/
│   ├── main.py                         # FastAPI app — all endpoints
│   ├── config.py                       # Centralized settings & language mapping
│   │
│   ├── models/
│   │   └── schemas.py                  # Pydantic request/response models (exact spec)
│   │
│   ├── transcription/
│   │   ├── audio_utils.py              # Base64 decode, chunking, temp files
│   │   ├── sarvam_service.py           # Sarvam AI Saaras v3 STT integration
│   │   └── whisper_fallback.py         # Groq Whisper backup ASR
│   │
│   ├── analysis/
│   │   ├── prompts.py                  # Master analysis prompt
│   │   ├── llm_service.py             # Multi-provider LLM with auto-fallback
│   │   └── sop_validator.py           # SOP validation + analytics + keywords
│   │
│   ├── vector_store/
│   │   └── chroma_store.py            # ChromaDB semantic search & storage
│   │
│   └── utils/
│       └── logger.py                   # Structured logging
│
├── tests/
│   └── test_api.py                     # API validation & schema tests
│
└── frontend/
    └── index.html                      # Interactive analytics dashboard
```

---

## Setup Instructions

### Prerequisites

- Python 3.12+
- FFmpeg (for audio processing)
- API keys: Sarvam AI, Google Gemini, Groq (all free tier)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/callsense-ai.git
cd callsense-ai
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
```

### 2. Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
winget install ffmpeg
# Or download from https://ffmpeg.org/download.html
```

### 3. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your keys
```

| Key | Where to Get (Free) |
|:----|:--------------------|
| `SARVAM_API_KEY` | [dashboard.sarvam.ai](https://dashboard.sarvam.ai) |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `API_SECRET_KEY` | Choose any secret key for API auth |

### 4. Run

```bash
# Development
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 5. Verify

```bash
# Health check
curl http://localhost:8000/health

# Open dashboard
# http://localhost:8000/dashboard
```

### 6. Docker

```bash
docker build -t callsense-ai .
docker run -p 8000:8000 --env-file .env callsense-ai
```

---

## API Usage

### Endpoint

```
POST /api/call-analytics
```

### Headers

```
Content-Type: application/json
x-api-key: YOUR_SECRET_API_KEY
```

### Request

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAA..."
}
```

### cURL Example

```bash
curl -X POST https://your-domain.com/api/call-analytics \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_track3_987654321" \
  -d '{
    "language": "Hindi",
    "audioFormat": "mp3",
    "audioBase64": "BASE64_ENCODED_AUDIO..."
  }'
```

### Response

```json
{
  "status": "success",
  "language": "Tamil",
  "transcript": "Agent: Hello, I am calling from Guvi Institution...",
  "summary": "Agent discussed Data Science course with customer...",
  "sop_validation": {
    "greeting": true,
    "identification": false,
    "problemStatement": true,
    "solutionOffering": true,
    "closing": true,
    "complianceScore": 0.8,
    "adherenceStatus": "NOT_FOLLOWED",
    "explanation": "Agent did not verify customer identity."
  },
  "analytics": {
    "paymentPreference": "EMI",
    "rejectionReason": "NONE",
    "sentiment": "Positive"
  },
  "keywords": ["Guvi", "Data Science", "EMI", "IIT Madras", "placement"]
}
```

### All Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/api/call-analytics` | Main analysis endpoint |
| `GET` | `/` | API info |
| `GET` | `/health` | Health check with provider status |
| `GET` | `/dashboard` | Interactive analytics dashboard |
| `GET` | `/api/search?q=query` | Semantic search across transcripts |
| `GET` | `/api/stats` | Vector store statistics |
| `GET` | `/docs` | Auto-generated API documentation (Swagger) |

---

## Approach

### Speech-to-Text Strategy

**Primary: Sarvam AI Saaras v3** — Trained on 1M+ hours of Indian audio, achieves 19.3% WER on IndicVoices benchmark. Outperforms GPT-4o Transcribe, Gemini 3 Pro, and Deepgram Nova3 on Indian languages. Natively handles Hinglish and Tanglish code-switching. Audio is split into 29-second chunks (Sarvam REST API limit), transcribed with `mode="translate"` for English output, then stitched together.

**Fallback: Groq Whisper Large v3** — If Sarvam fails (timeout, rate limit, error), the system automatically activates Groq's hosted Whisper model running at 216x real-time speed.

### Analysis Strategy

A single master prompt sends the full transcript to the LLM and extracts ALL required fields in one call: summary, SOP validation (5 boolean steps + compliance score + adherence status + explanation), payment preference, rejection reason, sentiment, and keywords.

**4-provider fallback chain** ensures the API never fails: Gemini 2.5 Flash → Groq Llama 3.3 70B → OpenRouter → NVIDIA NIMs.

### Validation Strategy

Every LLM output passes through a validation layer:

- **Pydantic Literal types** enforce exact enum values — invalid responses are physically impossible
- **Normalizers** auto-correct common LLM variations (e.g., "emi" → "EMI", "positive" → "Positive")
- **Keywords** are verified against the transcript text — hallucinated keywords are removed
- **Compliance score** is validated to be within 0.0-1.0 range
- **Adherence status** is derived from boolean fields: all true = "FOLLOWED", any false = "NOT_FOLLOWED"

---

## Edge Cases & Failure Handling

| Scenario | How We Handle It |
|:---------|:-----------------|
| Missing or invalid API key | Returns 401 Unauthorized immediately |
| Malformed base64 audio | Handles data URL prefixes, URL-safe encoding, padding issues |
| Audio longer than 30 seconds | Auto-chunks into 29s segments with 500ms overlap, stitches transcripts |
| Very short audio (<5s) | Processes as single chunk without splitting |
| Sarvam AI down/timeout | Auto-fallback to Groq Whisper Large v3 with retry logic |
| Gemini rate limited | Auto-fallback: Groq → OpenRouter → NVIDIA NIMs |
| LLM returns wrong enum value | Normalizer maps to closest valid enum (e.g., "installment" → "EMI") |
| LLM returns malformed JSON | Parser handles markdown fences, Python booleans, nested objects, extra text |
| Empty or garbage transcript | Returns valid response structure with default values and explanation |
| All ASR providers fail | Returns valid JSON with error message — never crashes, never returns invalid schema |
| All LLM providers fail | Keyword-based fallback analysis extracts SOP/payment/sentiment from transcript patterns |
| Concurrent requests | FastAPI async handles multiple requests, each gets unique call_id for tracking |
| Network interruption mid-chunk | Retry with exponential backoff (1s → 2s → 4s), max 3 retries per chunk |

---

## AI Tools Used

| Tool | Purpose | How Used |
|:-----|:--------|:---------|
| **Claude (Anthropic)** | Development assistance | Code scaffolding, architecture design, prompt engineering, debugging |
| **Sarvam AI Saaras v3** | Speech-to-Text | Production ASR engine — transcribes Hinglish/Tanglish call recordings |
| **Google Gemini 2.5 Flash** | NLP Analysis | Primary LLM — SOP validation, payment classification, summarization, sentiment |
| **Groq Llama 3.3 70B** | Fallback LLM | Secondary analysis engine activated when Gemini is unavailable |
| **Groq Whisper Large v3** | Fallback ASR | Backup transcription engine activated when Sarvam is unavailable |
| **OpenRouter** | Tertiary LLM | Additional fallback LLM provider for maximum reliability |
| **ChromaDB** | Vector Storage | Semantic indexing and search of call transcripts using embeddings |

> All AI tools are used as API services. No model training or fine-tuning was performed. All analysis is generated dynamically from the provided audio input — zero hardcoded responses.

---

## Performance

| Metric | Value |
|:-------|:------|
| Average response time | 15-45 seconds (depends on audio length + API latency) |
| Max audio duration | Up to 1 hour (chunked processing) |
| Supported audio formats | MP3, WAV, AAC, OGG, FLAC, M4A, WebM, AIFF, AMR, WMA |
| Languages | Hindi (Hinglish), Tamil (Tanglish) + 20 more Indian languages via Sarvam |
| Schema compliance | 100% — Pydantic Literal types make invalid responses impossible |

---

## Known Limitations

- Transcription accuracy depends on audio quality — background noise, speaker clarity, and phone line quality affect results
- SOP compliance detection relies on LLM interpretation which may vary on highly ambiguous or very short calls
- Processing time depends on external API response times (Sarvam AI, Google Gemini)
- ChromaDB uses default embedding model; domain-specific fine-tuned embeddings could improve search quality
- First request after cold start may be slightly slower due to model warm-up on provider side
- Compliance score calculation matches LLM assessment — edge cases may differ from manual human review

---

## Testing

```bash
# Run schema validation tests
python -m tests.test_api

# Test with a specific audio file
python -m tests.test_api path/to/audio.mp3 Hindi

# Test with sample audio
curl -o sample.mp3 "https://recordings.exotel.com/exotelrecordings/guvi64/5780094ea05a75c867120809da9a199f.mp3"
python -m tests.test_api sample.mp3 Tamil
```

---

<p align="center">
  <sub>Built for HCL GUVI Intern Hiring Hackathon 2026 — Track 3: Call Center Compliance</sub>
</p>
