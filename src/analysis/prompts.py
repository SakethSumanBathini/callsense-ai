"""
CallSense AI - LLM Prompts
The master analysis prompt that extracts ALL required fields from a transcript.
This is the most critical component - 60+ points depend on this prompt.
"""


MASTER_ANALYSIS_PROMPT = """You are an expert call center compliance analyst. Analyze the following call transcript and extract structured data.

IMPORTANT: The transcript may be in Hindi, Tamil, Hinglish, Tanglish, or English. Regardless of the transcript language, ALL your analysis and output MUST be in English. Translate and interpret the content to provide accurate English analysis.

TRANSCRIPT:
{transcript}

LANGUAGE: {language}

INSTRUCTIONS:
Analyze this call center conversation and return a JSON object with the following fields. Be precise and accurate.

1. SUMMARY: Write a concise 2-3 sentence summary capturing:
   - Who called whom and why
   - Key details discussed (amounts, products, dates)
   - The outcome of the call

2. SOP VALIDATION: Check if the agent followed the standard call center script.
   Evaluate each step:

   a) greeting (true/false): Did the agent start with a welcoming phrase?
      TRUE if: Agent says Hello, Hi, Namaste, Vanakkam, Good morning/afternoon/evening,
      Welcome, "calling from [company]", or any professional opening.
      FALSE if: Call starts abruptly without any greeting.

   b) identification (true/false): Did the agent ACTIVELY verify the customer's identity?
      TRUE if: Agent explicitly asks for or confirms customer's name, phone number,
      account number, policy number, registration ID, or any identity verification.
      FALSE if: Names are mentioned casually but no formal identity verification was done.
      IMPORTANT: Just calling someone by name is NOT identification. The agent must
      ASK FOR or VERIFY identity information.

   c) problemStatement (true/false): Was the purpose/problem of the call clearly stated?
      TRUE if: The reason for the call is discussed - customer's need, inquiry,
      complaint, follow-up, product interest, payment issue, etc.
      FALSE if: No clear purpose is established.

   d) solutionOffering (true/false): Did the agent offer a solution, product, or resolution?
      TRUE if: Agent proposes a course, plan, payment option, resolution, next steps,
      or any actionable solution to the customer's need.
      FALSE if: No solution or offering was presented.

   e) closing (true/false): Did the call end with a professional closing?
      TRUE if: Agent says Thank you, Thanks, Have a good day, Goodbye, or any
      professional sign-off.
      FALSE if: Call ends abruptly without closing.

   f) complianceScore: A float from 0.0 to 1.0 representing overall compliance.
      Consider the weight of each step:
      - Greeting present: +0.15
      - Identification done: +0.25 (most important for compliance)
      - Problem stated: +0.20
      - Solution offered: +0.25
      - Proper closing: +0.15
      Calculate based on which steps were followed. Round to 1 decimal.

   g) adherenceStatus: "FOLLOWED" if ALL 5 steps are true, "NOT_FOLLOWED" if ANY step is false.

   h) explanation: One sentence explaining what was followed and what was missed.

3. ANALYTICS:

   a) paymentPreference: Classify the payment intent. MUST be exactly one of:
      - "EMI" → Customer discusses installments, monthly payments, EMI plans
      - "FULL_PAYMENT" → Customer agrees to pay the full/total amount at once
      - "PARTIAL_PAYMENT" → Customer offers to pay part now, rest later
      - "DOWN_PAYMENT" → Customer pays an initial/advance amount to start a service
      If multiple are discussed, pick the one the customer prefers or agrees to.

   b) rejectionReason: If payment was NOT completed or customer declined, identify why.
      MUST be exactly one of:
      - "HIGH_INTEREST" → Customer complains about interest rates or charges being too high
      - "BUDGET_CONSTRAINTS" → Customer says they can't afford it, no money, tight budget
      - "ALREADY_PAID" → Customer claims payment was already made
      - "NOT_INTERESTED" → Customer explicitly declines or shows no interest
      - "NONE" → Payment was accepted/agreed OR no rejection occurred
      If the customer agreed to pay or showed interest, use "NONE".

   c) sentiment: Overall emotional tone of the conversation. MUST be exactly one of:
      - "Positive" → Customer is satisfied, agreeable, enthusiastic, or cooperative
      - "Negative" → Customer is frustrated, angry, complaining, or dissatisfied
      - "Neutral" → Matter-of-fact conversation, no strong emotions either way

4. KEYWORDS: Extract 5-15 important keywords/phrases from the transcript.
   Include: product/service names, company names, amounts/prices mentioned,
   technical terms, course names, payment-related terms, key topics discussed.
   Every keyword MUST appear in the transcript or be directly derivable from it.

RESPOND WITH ONLY THIS JSON (no markdown, no backticks, no explanation):
{{
  "summary": "...",
  "sop_validation": {{
    "greeting": true/false,
    "identification": true/false,
    "problemStatement": true/false,
    "solutionOffering": true/false,
    "closing": true/false,
    "complianceScore": 0.0,
    "adherenceStatus": "FOLLOWED or NOT_FOLLOWED",
    "explanation": "..."
  }},
  "analytics": {{
    "paymentPreference": "EMI or FULL_PAYMENT or PARTIAL_PAYMENT or DOWN_PAYMENT",
    "rejectionReason": "HIGH_INTEREST or BUDGET_CONSTRAINTS or ALREADY_PAID or NOT_INTERESTED or NONE",
    "sentiment": "Positive or Negative or Neutral"
  }},
  "keywords": ["keyword1", "keyword2", "..."]
}}"""


TRANSCRIPT_CLEANUP_PROMPT = """You are a transcript formatter. Clean up this raw speech-to-text output into a readable Agent/Customer dialogue format.

RAW TRANSCRIPT:
{raw_transcript}

RULES:
1. Format as "Agent: ..." and "Customer: ..." dialogue lines
2. Fix obvious speech-to-text errors (misheard words) based on context
3. DO NOT add any content that was not in the original
4. DO NOT remove any meaningful content
5. Keep proper nouns, numbers, and amounts exactly as they appear
6. If you cannot determine who is speaking, make your best inference based on context (the one selling/offering is the Agent, the one inquiring is the Customer)
7. Combine fragments into complete sentences where appropriate

Return ONLY the cleaned transcript, nothing else."""