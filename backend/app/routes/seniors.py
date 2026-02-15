"""Senior-related routes."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_current_user
from app.dependencies import get_checkins_collection
from app.models.auth import MeResponse
from app.models.checkin import CheckinDetail, CheckinListResponse, TriageStatus
from app.models.report import (
    BaselineMetric,
    BaselineResponse,
    WeeklySummary,
    WeeklySummaryRequest,
    WeeklySummaryResponse,
)
from app.models.alert import AlertResponse

router = APIRouter()

# In-memory storage for baselines and weekly summaries
BASELINES: Dict[str, List[BaselineMetric]] = {}
WEEKLY_SUMMARIES: Dict[str, List[WeeklySummary]] = {}

# Reference the ALERTS from alerts module
from app.routes.alerts import ALERTS


def _triage_status_from_db(value: Optional[str]) -> Optional[TriageStatus]:
    """Convert database triage value to enum."""
    if not value:
        return None
    lowered = value.lower()
    if lowered == "green":
        return TriageStatus.GREEN
    if lowered == "yellow":
        return TriageStatus.YELLOW
    if lowered == "red":
        return TriageStatus.RED
    return None


def _parse_date_filter(
    from_date: Optional[str], to_date: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Parse date range filter for MongoDB query."""
    if not from_date and not to_date:
        return None
    date_filter: Dict[str, Any] = {}
    if from_date:
        date_filter["$gte"] = datetime.fromisoformat(from_date)
    if to_date:
        date_filter["$lte"] = datetime.fromisoformat(to_date)
    return date_filter


def _checkin_detail_from_doc(doc: dict) -> CheckinDetail:
    """Convert MongoDB document to CheckinDetail model."""
    return CheckinDetail(
        checkin_id=doc.get("checkin_id", ""),
        senior_id=str(doc.get("user_id", "")),
        status=doc.get("status", "unknown"),
        started_at=doc.get("started_at", datetime.utcnow()),
        completed_at=doc.get("completed_at"),
        triage_status=_triage_status_from_db(doc.get("triage_status")),
        triage_reasons=doc.get("triage_reasons", []),
        transcript=doc.get("transcript"),
        facial_symmetry=doc.get("facial_symmetry_raw"),
        heart_rate=doc.get("heart_rate_raw"),
        camera_snapshot=doc.get("camera_snapshot"),
        ai_assessment=doc.get("ai_assessment"),
    )


@router.get("/{senior_id}/checkins", response_model=CheckinListResponse)
def list_checkins(
    senior_id: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    include_incomplete: bool = False,
) -> CheckinListResponse:
    """
    List all check-ins for a senior.
    
    Args:
        senior_id: The senior's user ID
        from_date: Optional start date filter (ISO format)
        to_date: Optional end date filter (ISO format)
        include_incomplete: If False (default), only returns completed check-ins with triage status
    """
    items: List[CheckinDetail] = []

    try:
        user_id = ObjectId(senior_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid senior_id format")

    query: Dict[str, Any] = {"user_id": user_id}
    
    # Filter out incomplete check-ins by default
    if not include_incomplete:
        query["status"] = "completed"
        query["triage_status"] = {"$ne": None}
    
    date_filter = _parse_date_filter(from_date, to_date)
    if date_filter:
        query["completed_at"] = date_filter

    docs = get_checkins_collection().find(query).sort("completed_at", -1)
    for doc in docs:
        items.append(_checkin_detail_from_doc(doc))

    return CheckinListResponse(senior_id=senior_id, items=items)


@router.get("/{senior_id}/baseline", response_model=BaselineResponse)
def get_baseline(senior_id: str) -> BaselineResponse:
    """Get baseline metrics for a senior."""
    metrics = BASELINES.get(senior_id, [])
    return BaselineResponse(senior_id=senior_id, metrics=metrics)


@router.post("/{senior_id}/summaries/weekly", response_model=WeeklySummaryResponse)
def create_weekly_summary(
    senior_id: str, payload: WeeklySummaryRequest
) -> WeeklySummaryResponse:
    """Create a weekly summary for a senior."""
    week_start = payload.week_start or datetime.utcnow().strftime("%Y-%m-%d")
    week_end = payload.week_start or datetime.utcnow().strftime("%Y-%m-%d")
    summary = WeeklySummary(
        summary_id=str(uuid4()),
        senior_id=senior_id,
        week_start=week_start,
        week_end=week_end,
        summary_text="Summary not generated yet.",
        key_trends=[],
        created_at=datetime.utcnow(),
    )
    WEEKLY_SUMMARIES.setdefault(senior_id, []).append(summary)
    return WeeklySummaryResponse(senior_id=senior_id, summary=summary)


@router.get("/{senior_id}/summaries/weekly", response_model=WeeklySummaryResponse)
def get_weekly_summary(
    senior_id: str, week_start: Optional[str] = None
) -> WeeklySummaryResponse:
    """Get the latest weekly summary for a senior."""
    summaries = WEEKLY_SUMMARIES.get(senior_id, [])
    if not summaries:
        raise HTTPException(status_code=404, detail="No weekly summaries available")

    if week_start:
        for summary in summaries:
            if summary.week_start == week_start:
                return WeeklySummaryResponse(senior_id=senior_id, summary=summary)
        raise HTTPException(status_code=404, detail="Weekly summary not found")

    return WeeklySummaryResponse(senior_id=senior_id, summary=summaries[-1])


@router.get("/{senior_id}/alerts", response_model=List[AlertResponse])
def list_alerts(senior_id: str) -> List[AlertResponse]:
    """List all alerts for a senior."""
    return ALERTS.get(senior_id, [])
