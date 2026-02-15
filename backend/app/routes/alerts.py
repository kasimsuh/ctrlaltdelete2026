"""Alert management routes."""

from datetime import datetime
from typing import Dict, List
from uuid import uuid4

from fastapi import APIRouter

from app.models.alert import AlertRequest, AlertResponse

router = APIRouter()

# In-memory storage for alerts
ALERTS: Dict[str, List[AlertResponse]] = {}


@router.post("/test", response_model=AlertResponse)
def test_alert(payload: AlertRequest) -> AlertResponse:
    """Create a test alert (for development purposes)."""
    alert = AlertResponse(
        alert_id=str(uuid4()),
        senior_id=payload.senior_id,
        level=payload.level,
        channel=payload.channel,
        target=payload.target,
        status="queued",
        created_at=datetime.utcnow(),
    )
    ALERTS.setdefault(payload.senior_id, []).append(alert)
    return alert
