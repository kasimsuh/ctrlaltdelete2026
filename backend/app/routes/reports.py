"""Report generation routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_current_user
from app.models.report import SeniorReportSummaryRequest, SeniorReportSummaryResponse
from app.services.ai_summary import generate_ai_summary

router = APIRouter()


def _require_doctor(user: Optional[dict]) -> dict:
    """Verify user has doctor role."""
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctor role required")
    return user


@router.post("/senior-summary", response_model=SeniorReportSummaryResponse)
def generate_senior_summary(
    payload: SeniorReportSummaryRequest,
    user: Optional[dict] = Depends(require_current_user),
) -> SeniorReportSummaryResponse:
    """Generate an AI-powered summary report for a senior's check-in history."""
    _require_doctor(user)
    return generate_ai_summary(payload)
