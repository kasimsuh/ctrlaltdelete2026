"""ElevenLabs agent helper routes + simple QA JSON persistence."""

from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_current_user

router = APIRouter()


class ElevenLabsSignedUrlResponse(BaseModel):
    signed_url: str


class QAPair(BaseModel):
    question: str
    answer: str


class QASaveRequest(BaseModel):
    # Deprecated: kept for backward-compat with older frontend builds.
    email: Optional[str] = None
    items: List[QAPair]


class QASaveResponse(BaseModel):
    path: str


@router.get("/elevenlabs/signed-url", response_model=ElevenLabsSignedUrlResponse)
def elevenlabs_signed_url(agent_id: Optional[str] = None) -> ElevenLabsSignedUrlResponse:
    """
    Return a short-lived signed URL for starting an ElevenLabs Conversational AI session
    from the browser without exposing the API key.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY is not set")

    agent = agent_id or os.getenv("ELEVENLABS_AGENT_ID")
    if not agent:
        raise HTTPException(status_code=500, detail="ELEVENLABS_AGENT_ID is not set")

    qs = urlencode({"agent_id": agent})
    url = f"https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?{qs}"
    req = Request(
        url,
        headers={
            "xi-api-key": api_key,
            "accept": "application/json",
            "user-agent": "guardian-checkin/0.1 (signed-url)",
        },
    )
    try:
        # Some macOS Python builds do not have system roots configured.
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urlopen(req, timeout=30, context=ctx) as resp:
            body = resp.read()
    except HTTPError as e:
        # HTTPError is also a file-like response.
        try:
            raw = e.read()
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        detail = f"ElevenLabs error {getattr(e, 'code', 'unknown')}"
        if text:
            detail += f": {text[:500]}"
        raise HTTPException(status_code=502, detail=detail)
    except URLError as e:
        raise HTTPException(
            status_code=502,
            detail=f"ElevenLabs request failed: URLError: {getattr(e, 'reason', '')}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"ElevenLabs request failed: {e.__class__.__name__}: {e}",
        )

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=502, detail="ElevenLabs signed-url response was not JSON")

    signed_url = (
        (payload or {}).get("signed_url")
        or (payload or {}).get("signedUrl")
        or (payload or {}).get("url")
    )
    if not signed_url:
        raise HTTPException(
            status_code=502,
            detail=f"ElevenLabs signed-url response missing URL field. Keys: {sorted((payload or {}).keys())}",
        )
    return ElevenLabsSignedUrlResponse(signed_url=signed_url)


@router.post("/stt/qa/save", response_model=QASaveResponse)
def save_qa_json(
    payload: QASaveRequest,
    user: Optional[dict] = Depends(require_current_user),
) -> QASaveResponse:
    """
    Overwrite a single JSON file containing only:
      - user_id
      - items: [{question, answer}]
    """
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(exist_ok=True)
    path = logs_dir / "stt.json"

    body = {
        "user_id": str(user.get("_id")),
        "items": [{"question": i.question, "answer": i.answer} for i in payload.items],
    }
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return QASaveResponse(path=str(path))
