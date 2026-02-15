"""Screening-related Pydantic models."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ScreeningResponseItem(BaseModel):
    q: str
    answer: Optional[bool] = None
    transcript: Optional[str] = None


class ScreeningSession(BaseModel):
    session_id: str
    senior_id: str
    checkin_id: Optional[str] = None
    timestamp: datetime
    responses: List[ScreeningResponseItem]


class ScreeningCreateRequest(BaseModel):
    session_id: Optional[str] = None
    checkin_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    responses: List[ScreeningResponseItem]


class ScreeningCreateResponse(BaseModel):
    session_id: str
    stored_at: datetime
