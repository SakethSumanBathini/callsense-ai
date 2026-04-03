"""
CallSense AI - Request/Response Models
Pydantic models enforcing EXACT schema required by HCL GUVI evaluation.
Every field name, type, and enum value matches the specification precisely.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional


# =============================================================================
# REQUEST MODEL
# =============================================================================

class CallAnalyticsRequest(BaseModel):
    """
    Incoming request body - matches exact specification.
    Fields: language, audioFormat, audioBase64
    """
    language: str = Field(
        ...,
        description="Language of the audio: Tamil (Tanglish) or Hindi (Hinglish)",
        examples=["Tamil", "Hindi"]
    )
    audioFormat: str = Field(
        default="mp3",
        description="Audio format - always mp3",
        examples=["mp3"]
    )
    audioBase64: str = Field(
        ...,
        description="Base64-encoded MP3 audio of the call recording",
        min_length=10
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        valid = ["Tamil", "Hindi", "English", "Bengali", "Telugu", 
                 "Kannada", "Malayalam", "Marathi", "Gujarati", "Punjabi", "Odia"]
        if v not in valid:
            # Try case-insensitive match
            for lang in valid:
                if v.lower() == lang.lower():
                    return lang
            return "Hindi"  # Safe default
        return v


# =============================================================================
# RESPONSE MODELS - EXACT SCHEMA FROM SPECIFICATION
# =============================================================================

class SOPValidation(BaseModel):
    """
    SOP Validation object - EXACT field names from spec.
    greeting, identification, problemStatement, solutionOffering, closing
    complianceScore (0.0-1.0), adherenceStatus, explanation
    """
    greeting: bool = Field(
        ..., description="Whether agent used a proper greeting"
    )
    identification: bool = Field(
        ..., description="Whether agent verified customer identity"
    )
    problemStatement: bool = Field(
        ..., description="Whether the purpose/problem was clearly stated"
    )
    solutionOffering: bool = Field(
        ..., description="Whether agent offered a solution/product/plan"
    )
    closing: bool = Field(
        ..., description="Whether call ended with professional closing"
    )
    complianceScore: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall compliance score from 0.0 to 1.0"
    )
    adherenceStatus: Literal["FOLLOWED", "NOT_FOLLOWED"] = Field(
        ..., description="Whether full SOP was followed"
    )
    explanation: str = Field(
        ..., description="Brief explanation of compliance assessment"
    )

    @field_validator("complianceScore")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 1)


class Analytics(BaseModel):
    """
    Analytics object - EXACT enum values from spec.
    Payment: EMI | FULL_PAYMENT | PARTIAL_PAYMENT | DOWN_PAYMENT
    Rejection: HIGH_INTEREST | BUDGET_CONSTRAINTS | ALREADY_PAID | NOT_INTERESTED | NONE
    Sentiment: Positive | Negative | Neutral
    """
    paymentPreference: Literal[
        "EMI", "FULL_PAYMENT", "PARTIAL_PAYMENT", "DOWN_PAYMENT"
    ] = Field(
        ..., description="Classified payment preference"
    )
    rejectionReason: Literal[
        "HIGH_INTEREST", "BUDGET_CONSTRAINTS", "ALREADY_PAID",
        "NOT_INTERESTED", "NONE"
    ] = Field(
        ..., description="Reason for rejection if payment not completed"
    )
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        ..., description="Overall call sentiment"
    )


class CallAnalyticsResponse(BaseModel):
    """
    Complete response - EXACT structure from specification.
    Any deviation = 0 points for that test case.
    """
    status: Literal["success", "error"] = Field(
        default="success", description="Response status"
    )
    language: str = Field(
        ..., description="Detected/specified language"
    )
    transcript: str = Field(
        ..., description="Full speech-to-text output"
    )
    summary: str = Field(
        ..., description="AI-powered summary of the conversation"
    )
    sop_validation: SOPValidation = Field(
        ..., description="SOP compliance validation results"
    )
    analytics: Analytics = Field(
        ..., description="Business analytics extraction"
    )
    keywords: List[str] = Field(
        ..., description="Main keywords from transcript and summary",
        min_length=1
    )


class ErrorResponse(BaseModel):
    """Error response format."""
    status: Literal["error"] = "error"
    message: str = ""
    language: str = ""
    transcript: str = ""
    summary: str = ""
    sop_validation: dict = {
        "greeting": False,
        "identification": False,
        "problemStatement": False,
        "solutionOffering": False,
        "closing": False,
        "complianceScore": 0.0,
        "adherenceStatus": "NOT_FOLLOWED",
        "explanation": "Error processing audio"
    }
    analytics: dict = {
        "paymentPreference": "EMI",
        "rejectionReason": "NONE",
        "sentiment": "Neutral"
    }
    keywords: List[str] = ["error"]
