"""Gemini assessment for a single check-in using structured inputs + ElevenLabs STT Q/A."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from google import genai

logger = logging.getLogger("guardian")


def _extract_json(raw_text: str) -> Optional[str]:
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return None
    return cleaned[json_start : json_end + 1]


def _fallback(model: str, note: str) -> dict[str, Any]:
    return {
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": "AI assessment unavailable.",
        "symptoms": [],
        "risks": [],
        "follow_up": [],
        "signals": {},
        "note": note,
    }


def assess_checkin_with_stt(
    *,
    stt_items: list[dict[str, Any]],
    answers: dict[str, Any],
    transcript: Optional[str],
    facial_symmetry: Optional[dict[str, Any]],
    heart_rate: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Use Gemini to interpret speech Q/A safely (handle negations), and produce a concise,
    structured summary suitable for clinician review.

    Returns a dict with keys:
      - summary: str
      - symptoms: list[str]
      - risks: list[str]
      - follow_up: list[str]
      - signals: dict[str, str] where values are "present" | "absent" | "unclear"
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_CHECKIN_MODEL") or "gemini-2.0-flash"
    if not api_key:
        return _fallback(model, "GEMINI_API_KEY is not set")

    # Keep prompt payload small and deterministic.
    prompt_payload = {
        "instructions": {
            "task": "Summarize and extract clinically relevant signals from a single check-in.",
            "critical_rule": (
                "Do not infer symptoms from substring mentions. Treat negations explicitly "
                "(e.g., 'no dizziness' => dizziness=absent). If ambiguous, mark as unclear."
            ),
        },
        "canonical_questions": [
            "How are you feeling today?",
            "Are you experiencing any dizziness, chest pain, or trouble breathing?",
            "Did you take your morning medications?",
        ],
        "structured_answers": answers,
        "speech_qa": [
            {
                "index": idx + 1,
                "question": str(i.get("question") or ""),
                "answer": str(i.get("answer") or ""),
            }
            for idx, i in enumerate(stt_items or [])
        ],
        "screening_transcript": transcript or "",
        "facial_symmetry": facial_symmetry or None,
        "heart_rate": heart_rate or None,
        "output_schema": {
            "summary": "string",
            "symptoms": ["string"],
            "risks": ["string"],
            "follow_up": ["string"],
            "signals": {
                "dizziness": "present|absent|unclear",
                "chest_pain": "present|absent|unclear",
                "trouble_breathing": "present|absent|unclear",
                "medication_missed": "present|absent|unclear",
            },
        },
    }

    prompt = (
        "You are a clinical assistant helping a doctor review a single senior check-in.\n"
        "Return ONLY a JSON object that matches output_schema.\n"
        "Rules:\n"
        "1) Do not diagnose.\n"
        "2) Do not assume symptoms are present because a word appears; handle negations.\n"
        "3) Use 'unclear' when the patient did not clearly confirm or deny.\n"
        "4) Be concise.\n\n"
        f"DATA:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
    )

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception as e:
        logger.exception("Gemini check-in assessment failed")
        return _fallback(model, f"Gemini request failed: {e.__class__.__name__}")

    raw_text = (getattr(response, "text", None) or "").strip()
    json_text = _extract_json(raw_text)
    if not json_text:
        return _fallback(model, "Gemini response was not JSON")

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        logger.exception("Gemini check-in assessment JSON parse failed")
        return _fallback(model, "Gemini response JSON parse failed")

    # Normalize and return.
    return {
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": str(parsed.get("summary") or "").strip() or "Limited data.",
        "symptoms": list(parsed.get("symptoms") or []),
        "risks": list(parsed.get("risks") or []),
        "follow_up": list(parsed.get("follow_up") or []),
        "signals": dict(parsed.get("signals") or {}),
    }
