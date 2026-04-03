"""
CallSense AI - API Tests
Test the API endpoint with sample audio and validate response schema.
"""

import json
import base64
import httpx
import sys
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_SECRET_KEY", "sk_track3_987654321")

# Valid enum values from competition spec
VALID_PAYMENTS = ["EMI", "FULL_PAYMENT", "PARTIAL_PAYMENT", "DOWN_PAYMENT"]
VALID_REJECTIONS = [
    "HIGH_INTEREST", "BUDGET_CONSTRAINTS", "ALREADY_PAID",
    "NOT_INTERESTED", "NONE",
]
VALID_SENTIMENTS = ["Positive", "Negative", "Neutral"]
VALID_ADHERENCE = ["FOLLOWED", "NOT_FOLLOWED"]


def validate_response(response_data: dict) -> list:
    """
    Validate response matches exact competition schema.
    Returns list of errors (empty = perfect).
    """
    errors = []

    # Top-level fields
    required_fields = [
        "status", "language", "transcript", "summary",
        "sop_validation", "analytics", "keywords",
    ]
    for field in required_fields:
        if field not in response_data:
            errors.append(f"Missing top-level field: {field}")

    # Status
    if response_data.get("status") not in ["success", "error"]:
        errors.append(f"Invalid status: {response_data.get('status')}")

    # Transcript
    if not response_data.get("transcript"):
        errors.append("Transcript is empty")

    # Summary
    if not response_data.get("summary"):
        errors.append("Summary is empty")

    # SOP Validation
    sop = response_data.get("sop_validation", {})
    sop_booleans = [
        "greeting", "identification", "problemStatement",
        "solutionOffering", "closing",
    ]
    for field in sop_booleans:
        if field not in sop:
            errors.append(f"Missing SOP field: {field}")
        elif not isinstance(sop[field], bool):
            errors.append(f"SOP {field} is not boolean: {type(sop[field])}")

    if "complianceScore" not in sop:
        errors.append("Missing complianceScore")
    elif not isinstance(sop["complianceScore"], (int, float)):
        errors.append(f"complianceScore not numeric: {type(sop['complianceScore'])}")
    elif not (0.0 <= sop["complianceScore"] <= 1.0):
        errors.append(f"complianceScore out of range: {sop['complianceScore']}")

    if sop.get("adherenceStatus") not in VALID_ADHERENCE:
        errors.append(f"Invalid adherenceStatus: {sop.get('adherenceStatus')}")

    if not sop.get("explanation"):
        errors.append("Missing explanation")

    # Analytics
    analytics = response_data.get("analytics", {})
    if analytics.get("paymentPreference") not in VALID_PAYMENTS:
        errors.append(
            f"Invalid paymentPreference: {analytics.get('paymentPreference')}"
        )
    if analytics.get("rejectionReason") not in VALID_REJECTIONS:
        errors.append(
            f"Invalid rejectionReason: {analytics.get('rejectionReason')}"
        )
    if analytics.get("sentiment") not in VALID_SENTIMENTS:
        errors.append(
            f"Invalid sentiment: {analytics.get('sentiment')}"
        )

    # Keywords
    keywords = response_data.get("keywords", [])
    if not isinstance(keywords, list):
        errors.append(f"keywords is not a list: {type(keywords)}")
    elif len(keywords) == 0:
        errors.append("keywords array is empty")

    return errors


def test_with_audio_file(audio_path: str, language: str = "Hindi"):
    """Test API with an actual audio file."""
    print(f"\n{'='*60}")
    print(f"Testing with: {audio_path}")
    print(f"Language: {language}")
    print(f"{'='*60}")

    # Read and encode audio
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Build request
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64,
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
    }

    # Send request
    print("Sending request...")
    try:
        response = httpx.post(
            f"{API_URL}/api/call-analytics",
            json=payload,
            headers=headers,
            timeout=180.0,
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            # Validate schema
            errors = validate_response(data)

            if errors:
                print(f"\n❌ SCHEMA ERRORS ({len(errors)}):")
                for e in errors:
                    print(f"  - {e}")
            else:
                print("\n✅ SCHEMA VALID - All fields correct!")

            # Print response
            print(f"\nTranscript length: {len(data.get('transcript', ''))}")
            print(f"Summary: {data.get('summary', '')[:100]}...")
            print(f"SOP: {json.dumps(data.get('sop_validation', {}), indent=2)}")
            print(f"Analytics: {json.dumps(data.get('analytics', {}), indent=2)}")
            print(f"Keywords: {data.get('keywords', [])}")

            return data
        else:
            print(f"Error: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"Request failed: {e}")
        return None


def test_auth():
    """Test that API key authentication works."""
    print("\n--- Testing Authentication ---")

    # Test without key
    try:
        r = httpx.post(
            f"{API_URL}/api/call-analytics",
            json={"language": "Hindi", "audioFormat": "mp3", "audioBase64": "test"},
            timeout=10.0,
        )
        assert r.status_code == 422 or r.status_code == 401, \
            f"Expected 401/422 without key, got {r.status_code}"
        print("✅ No key → rejected correctly")
    except Exception as e:
        print(f"⚠️ Auth test error: {e}")

    # Test with wrong key
    try:
        r = httpx.post(
            f"{API_URL}/api/call-analytics",
            json={"language": "Hindi", "audioFormat": "mp3", "audioBase64": "test"},
            headers={"x-api-key": "wrong_key"},
            timeout=10.0,
        )
        assert r.status_code == 401, \
            f"Expected 401 with wrong key, got {r.status_code}"
        print("✅ Wrong key → rejected correctly")
    except Exception as e:
        print(f"⚠️ Auth test error: {e}")


def test_health():
    """Test health endpoint."""
    print("\n--- Testing Health ---")
    try:
        r = httpx.get(f"{API_URL}/health", timeout=10.0)
        print(f"Health: {r.json()}")
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Health check failed: {e}")


if __name__ == "__main__":
    test_health()
    test_auth()

    # Test with sample audio if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else "Hindi"
        test_with_audio_file(audio_file, language)
    else:
        print("\nUsage: python -m tests.test_api <audio_file.mp3> [Hindi|Tamil]")
        print("Or just run to test health and auth.")
