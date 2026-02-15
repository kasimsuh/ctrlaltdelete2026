"""AI summary generation service using Gemini."""

import json
import logging
import os
from typing import Optional

from google import genai

from app.models.report import SeniorReportSummaryRequest, SeniorReportSummaryResponse

logger = logging.getLogger("guardian")


def _build_prompt_payload(payload: SeniorReportSummaryRequest) -> dict:
    """Build the data payload for the AI prompt."""
    return {
        "senior": {
            "id": payload.senior_id,
            "name": payload.senior_name,
            "email": payload.senior_email,
        },
        "overview": payload.overview.model_dump(),
        "recent_checkins": [item.model_dump() for item in payload.recent_checkins],
    }


def _build_prompt(prompt_payload: dict) -> str:
    """Build the AI prompt text."""
    return (
        "You are a clinical assistant helping a doctor review a senior's check-in history.\n"
        "Return ONLY a JSON object with exactly these keys: summary, symptoms, risks, follow_up.\n"
        "summary is a short sentence. symptoms, risks, follow_up are arrays of strings.\n"
        "No markdown, no extra text, no code fences. Be concise, no diagnosis, "
        "no speculation beyond the data. If data is limited, set summary to 'Limited data.' "
        "and keep arrays short.\n\n"
        f"DATA:\n{json.dumps(prompt_payload, indent=2)}"
    )


def _extract_json(raw_text: str) -> Optional[str]:
    """Extract JSON from potentially formatted AI response."""
    cleaned = raw_text.strip()
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


def _empty_response() -> SeniorReportSummaryResponse:
    """Return an empty/fallback summary response."""
    return SeniorReportSummaryResponse(
        summary="AI summary unavailable.",
        symptoms=[],
        risks=[],
        follow_up=[],
    )


def generate_ai_summary(
    payload: SeniorReportSummaryRequest,
) -> SeniorReportSummaryResponse:
    """Generate an AI summary for a senior's check-in history."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key missing for AI summary")
        return _empty_response()

    client = genai.Client(api_key=api_key)
    prompt_payload = _build_prompt_payload(payload)
    prompt = _build_prompt(prompt_payload)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
    except Exception:
        logger.exception("Failed to generate Gemini summary")
        return _empty_response()

    raw_text = (getattr(response, "text", None) or "").strip()
    if not raw_text:
        logger.warning("Gemini summary empty for senior_id=%s", payload.senior_id)
        return _empty_response()

    json_text = _extract_json(raw_text)
    if not json_text:
        logger.warning("Gemini summary not JSON for senior_id=%s", payload.senior_id)
        return _empty_response()

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        logger.exception("Gemini summary JSON parse failed")
        return _empty_response()

    return SeniorReportSummaryResponse(
        summary=str(parsed.get("summary", "AI summary unavailable.")),
        symptoms=list(parsed.get("symptoms", []) or []),
        risks=list(parsed.get("risks", []) or []),
        follow_up=list(parsed.get("follow_up", []) or []),
    )
