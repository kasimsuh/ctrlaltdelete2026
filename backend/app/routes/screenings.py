"""Screening session routes."""

from datetime import datetime
from typing import Dict
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.dependencies import get_checkins_collection, get_screenings_collection
from app.models.screening import (
    ScreeningCreateRequest,
    ScreeningCreateResponse,
    ScreeningSession,
)
from app.services.screening import build_screening_transcript

router = APIRouter()

# In-memory storage for screening sessions
SCREENINGS: Dict[str, ScreeningSession] = {}


@router.post("", response_model=ScreeningCreateResponse)
def create_screening(payload: ScreeningCreateRequest) -> ScreeningCreateResponse:
    """Create a new screening session."""
    if not payload.checkin_id:
        raise HTTPException(status_code=400, detail="checkin_id is required")

    session_id = payload.session_id or f"screening-{uuid4()}"
    timestamp = payload.timestamp or datetime.utcnow()

    # Get checkin to find the user
    checkin_doc = get_checkins_collection().find_one({"checkin_id": payload.checkin_id})
    if not checkin_doc:
        raise HTTPException(status_code=404, detail="Check-in not found")

    senior_id = str(checkin_doc.get("user_id", ""))
    session = ScreeningSession(
        session_id=session_id,
        senior_id=senior_id,
        checkin_id=payload.checkin_id,
        timestamp=timestamp,
        responses=payload.responses,
    )
    SCREENINGS[session_id] = session

    get_screenings_collection().insert_one(
        {
            "session_id": session_id,
            "senior_id": senior_id,
            "checkin_id": payload.checkin_id,
            "timestamp": timestamp,
            "responses": [item.model_dump() for item in payload.responses],
            "transcript": build_screening_transcript(payload.responses),
        }
    )

    if payload.checkin_id:
        get_checkins_collection().update_one(
            {"checkin_id": payload.checkin_id},
            {
                "$set": {
                    "screening_session_id": session_id,
                    "screening_responses": [
                        item.model_dump() for item in payload.responses
                    ],
                    "transcript": build_screening_transcript(payload.responses),
                }
            },
        )
    return ScreeningCreateResponse(session_id=session_id, stored_at=datetime.utcnow())


@router.get("/{session_id}", response_model=ScreeningSession)
def get_screening(session_id: str) -> ScreeningSession:
    """Get a screening session by ID."""
    session = SCREENINGS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Screening session not found")
    return session
