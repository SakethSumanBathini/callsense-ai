"""
CallSense AI - Extended Analytics Engine
Adds enterprise-grade features beyond basic SOP compliance:
- Multi-dimensional scoring (empathy, professionalism, active listening)
- Risk/escalation detection
- Agent coaching recommendations
- Talk pattern analysis
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def compute_talk_patterns(transcript: str) -> Dict[str, Any]:
    """
    Analyze talk patterns from transcript text.
    Computes agent vs customer word counts, turn counts, and ratios.
    """
    agent_words = 0
    customer_words = 0
    agent_turns = 0
    customer_turns = 0
    
    lines = transcript.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect speaker from common patterns
        lower = line.lower()
        if lower.startswith('agent:') or lower.startswith('agent ') or lower.startswith('caller:'):
            words = len(line.split()) - 1  # subtract label
            agent_words += max(words, 0)
            agent_turns += 1
        elif lower.startswith('customer:') or lower.startswith('customer ') or lower.startswith('user:'):
            words = len(line.split()) - 1
            customer_words += max(words, 0)
            customer_turns += 1
        else:
            # If no speaker label, count as general
            agent_words += len(line.split()) // 2
            customer_words += len(line.split()) // 2
    
    # If no labeled lines found, estimate from total
    total_words = agent_words + customer_words
    if total_words == 0:
        total_words = len(transcript.split())
        agent_words = total_words * 6 // 10  # agents typically talk more
        customer_words = total_words - agent_words
        agent_turns = max(1, total_words // 50)
        customer_turns = max(1, total_words // 60)
    
    total = max(agent_words + customer_words, 1)
    
    return {
        "agent_word_count": agent_words,
        "customer_word_count": customer_words,
        "agent_talk_ratio": round(agent_words / total, 2),
        "customer_talk_ratio": round(customer_words / total, 2),
        "agent_turns": agent_turns,
        "customer_turns": customer_turns,
        "avg_agent_turn_length": round(agent_words / max(agent_turns, 1), 1),
        "avg_customer_turn_length": round(customer_words / max(customer_turns, 1), 1),
        "total_words": total_words,
        "ideal_ratio_benchmark": "43% agent / 57% customer (Gong.io research)"
    }


def detect_risk_signals(transcript: str, sentiment: str) -> Dict[str, Any]:
    """
    Detect risk and escalation signals in the call.
    Combines keyword triggers with sentiment analysis.
    """
    text_lower = transcript.lower()
    
    # Risk keyword categories
    risk_keywords = {
        "cancellation": ["cancel", "terminate", "stop service", "close account", "discontinue", "unsubscribe"],
        "escalation": ["supervisor", "manager", "complaint", "escalate", "higher authority", "legal"],
        "competitor": ["competitor", "other company", "switching", "better offer", "alternative"],
        "dissatisfaction": ["worst", "terrible", "horrible", "unacceptable", "disgusting", "fed up", "frustrated"],
        "threat": ["lawsuit", "legal action", "consumer court", "report", "social media"],
        "urgency": ["urgent", "immediately", "right now", "emergency", "critical"],
    }
    
    triggered = {}
    risk_score = 0
    triggers_found = []
    
    for category, keywords in risk_keywords.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            triggered[category] = found
            triggers_found.extend(found)
            risk_score += len(found) * 15  # Each trigger adds to score
    
    # Sentiment amplification
    if sentiment == "Negative":
        risk_score += 20
    elif sentiment == "Neutral":
        risk_score += 5
    
    # Cap at 100
    risk_score = min(risk_score, 100)
    
    # Determine risk level
    if risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "escalation_probability": round(min(risk_score / 100, 1.0), 2),
        "risk_triggers": triggers_found[:10],
        "risk_categories": list(triggered.keys()),
        "requires_attention": risk_score >= 40,
    }


def generate_coaching(
    sop_validation: Dict,
    analytics: Dict,
    talk_patterns: Dict,
    risk: Dict
) -> Dict[str, Any]:
    """
    Generate personalized coaching recommendations based on call analysis.
    """
    recommendations = []
    strengths = []
    improvements = []
    
    # SOP-based coaching
    if sop_validation.get("greeting"):
        strengths.append("Professional greeting established rapport")
    else:
        improvements.append("Opening greeting")
        recommendations.append(
            "Start every call with a warm greeting — 'Hello, thank you for calling [company]. "
            "My name is [name], how can I assist you today?' First impressions set the tone."
        )
    
    if sop_validation.get("identification"):
        strengths.append("Customer identity properly verified")
    else:
        improvements.append("Customer verification")
        recommendations.append(
            "Always verify customer identity before discussing account details. Ask: "
            "'Can I confirm your registered name and phone number?' This builds trust and ensures compliance."
        )
    
    if not sop_validation.get("problemStatement"):
        improvements.append("Problem clarification")
        recommendations.append(
            "Clearly restate the customer's problem before offering solutions. Use: "
            "'Let me make sure I understand — you are looking for...' Active listening reduces resolution time."
        )
    
    if sop_validation.get("solutionOffering"):
        strengths.append("Solution clearly presented to customer")
    else:
        improvements.append("Solution presentation")
        recommendations.append(
            "Present at least one clear solution with specific next steps. "
            "Customers respond better to concrete options than vague promises."
        )
    
    if sop_validation.get("closing"):
        strengths.append("Professional closing with clear next steps")
    else:
        improvements.append("Call closing")
        recommendations.append(
            "End every call with a summary of next steps and a professional closing. "
            "'Thank you for your time. I will [action]. Is there anything else I can help with?'"
        )
    
    # Talk pattern coaching
    agent_ratio = talk_patterns.get("agent_talk_ratio", 0.5)
    if agent_ratio > 0.65:
        recommendations.append(
            f"Your talk ratio is {int(agent_ratio*100)}% — aim for 43%. "
            "Try asking more open-ended questions and pausing after key points. "
            "Top performers listen more than they talk."
        )
        improvements.append("Talk-to-listen ratio (too much talking)")
    elif agent_ratio < 0.3:
        recommendations.append(
            f"Your talk ratio is only {int(agent_ratio*100)}% — the customer dominated the conversation. "
            "Take more initiative in guiding the discussion toward resolution."
        )
        improvements.append("Talk-to-listen ratio (too passive)")
    else:
        strengths.append(f"Good talk-to-listen ratio ({int(agent_ratio*100)}% agent)")
    
    # Risk coaching
    if risk.get("risk_level") == "HIGH":
        recommendations.append(
            "This call showed high risk signals. When customers express frustration, "
            "acknowledge their feelings first: 'I understand this is frustrating.' "
            "De-escalation before solution prevents churn."
        )
    
    # Ensure at least 3 recommendations
    if len(recommendations) < 3:
        generic = [
            "Summarize key points before ending the call to ensure mutual understanding.",
            "Use the customer's name 2-3 times during the conversation to build rapport.",
            "After explaining the solution, confirm understanding: 'Does that make sense? Do you have any questions?'",
        ]
        for g in generic:
            if len(recommendations) >= 3:
                break
            recommendations.append(g)
    
    # Overall performance rating
    score_factors = [
        1 if sop_validation.get("greeting") else 0,
        1 if sop_validation.get("identification") else 0,
        1 if sop_validation.get("problemStatement") else 0,
        1 if sop_validation.get("solutionOffering") else 0,
        1 if sop_validation.get("closing") else 0,
        1 if 0.35 <= agent_ratio <= 0.55 else 0,
        1 if risk.get("risk_level") != "HIGH" else 0,
    ]
    performance_rating = round(sum(score_factors) / len(score_factors) * 10, 1)
    
    return {
        "overall_performance": performance_rating,
        "recommendations": recommendations[:5],
        "strengths": strengths,
        "areas_for_improvement": improvements,
        "performance_label": (
            "Excellent" if performance_rating >= 8 else
            "Good" if performance_rating >= 6 else
            "Needs Improvement" if performance_rating >= 4 else
            "Critical - Requires Training"
        ),
    }


def compute_multi_dimensional_score(
    sop_validation: Dict,
    transcript: str,
    sentiment: str,
    talk_patterns: Dict
) -> Dict[str, Any]:
    """
    Multi-dimensional call scoring beyond basic SOP.
    Scores: empathy, professionalism, active listening, resolution quality.
    """
    text_lower = transcript.lower()
    
    # Empathy indicators
    empathy_phrases = [
        "i understand", "i appreciate", "sorry to hear", "i can see",
        "that must be", "don't worry", "no problem", "happy to help",
        "let me help", "i'm here to", "absolutely", "of course",
        "take your time", "no rush", "completely understand",
    ]
    empathy_count = sum(1 for p in empathy_phrases if p in text_lower)
    empathy_score = min(10, 3 + empathy_count * 1.5)
    
    # Professionalism indicators
    professional_phrases = [
        "thank you", "please", "sir", "ma'am", "madam",
        "certainly", "sure", "glad to", "welcome", "have a good",
    ]
    unprofessional_phrases = [
        "shut up", "stupid", "idiot", "whatever", "don't care",
    ]
    prof_count = sum(1 for p in professional_phrases if p in text_lower)
    unprof_count = sum(1 for p in unprofessional_phrases if p in text_lower)
    professionalism_score = min(10, max(2, 5 + prof_count - unprof_count * 3))
    
    # Active listening (questions asked, confirmations)
    question_count = text_lower.count('?')
    confirmation_phrases = ["i see", "okay", "right", "got it", "noted", "understood"]
    confirm_count = sum(1 for p in confirmation_phrases if p in text_lower)
    listening_score = min(10, 3 + question_count * 0.5 + confirm_count)
    
    # Resolution quality (based on SOP + outcome)
    sop_true_count = sum(1 for k in ['greeting', 'identification', 'problemStatement', 'solutionOffering', 'closing'] if sop_validation.get(k))
    resolution_base = sop_true_count * 2  # 0-10
    if sentiment == "Positive":
        resolution_base = min(10, resolution_base + 2)
    elif sentiment == "Negative":
        resolution_base = max(2, resolution_base - 2)
    resolution_score = resolution_base
    
    # Overall quality (weighted average)
    overall = round(
        empathy_score * 0.2 + 
        professionalism_score * 0.25 + 
        listening_score * 0.2 + 
        resolution_score * 0.35, 1
    )
    
    return {
        "empathy": round(empathy_score, 1),
        "professionalism": round(professionalism_score, 1),
        "active_listening": round(listening_score, 1),
        "resolution_quality": round(resolution_score, 1),
        "overall_quality": overall,
        "quality_label": (
            "Excellent" if overall >= 8 else
            "Good" if overall >= 6 else
            "Average" if overall >= 4 else
            "Below Standard"
        ),
    }


def build_extended_analytics(
    transcript: str,
    sop_validation: Dict,
    analytics: Dict,
) -> Dict[str, Any]:
    """
    Build the complete extended analytics package.
    This is the bonus data that goes beyond required fields.
    """
    try:
        # Compute talk patterns
        talk_patterns = compute_talk_patterns(transcript)
        
        # Detect risk signals
        sentiment = analytics.get("sentiment", "Neutral")
        risk = detect_risk_signals(transcript, sentiment)
        
        # Multi-dimensional scoring
        multi_score = compute_multi_dimensional_score(
            sop_validation, transcript, sentiment, talk_patterns
        )
        
        # Generate coaching
        coaching = generate_coaching(sop_validation, analytics, talk_patterns, risk)
        
        return {
            "multi_dimensional_score": multi_score,
            "risk_detection": risk,
            "talk_patterns": talk_patterns,
            "coaching": coaching,
        }
    except Exception as e:
        logger.error(f"Extended analytics failed: {e}")
        return {}
