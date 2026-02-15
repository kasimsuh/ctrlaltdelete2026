"""Dashboard routes for doctor view."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_current_user
from app.db import get_dashboard_analytics, get_latest_checkins, get_senior_users

router = APIRouter()


def _require_doctor(user: Optional[dict]) -> dict:
    """Verify user has doctor role."""
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctor role required")
    return user


def _serialize_datetime(value) -> Optional[str]:
    """Serialize datetime to ISO format string."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@router.get("/analytics")
def dashboard_analytics(user: Optional[dict] = Depends(require_current_user)):
    """Get dashboard analytics for the doctor."""
    _require_doctor(user)
    analytics = get_dashboard_analytics(days=7)
    print(f"[DASHBOARD] Analytics loaded for doctor {user.get('email')}: {analytics}")
    return analytics


@router.get("/seniors")
def dashboard_seniors(user: Optional[dict] = Depends(require_current_user)):
    """Get list of seniors with their latest check-in status."""
    _require_doctor(user)
    seniors = get_senior_users(limit=50)
    user_ids = [senior["_id"] for senior in seniors]
    latest_checkins = get_latest_checkins(user_ids)

    print(f"[DASHBOARD] Fetching seniors list for doctor {user.get('email')}")
    print(f"[DASHBOARD] Total seniors: {len(seniors)}")

    response_items = []
    for senior in seniors:
        last_checkin = latest_checkins.get(senior["_id"])
        
        # Extract heart rate if available
        heart_rate_data = None
        if last_checkin and last_checkin.get("heart_rate_raw"):
            hr_raw = last_checkin.get("heart_rate_raw")
            heart_rate_data = {
                "avg_hr_bpm": hr_raw.get("avg_hr_bpm"),
                "hr_quality": hr_raw.get("hr_quality", "low"),
            }
        
        item = {
            "id": str(senior["_id"]),
            "firstName": senior.get("firstName", ""),
            "lastName": senior.get("lastName", ""),
            "email": senior.get("email", ""),
            "lastCheckinAt": (
                _serialize_datetime(last_checkin.get("completed_at"))
                if last_checkin
                else None
            ),
            "triageStatus": (
                last_checkin.get("triage_status") if last_checkin else None
            ),
            "checkinId": (last_checkin.get("checkin_id") if last_checkin else None),
            "heartRate": heart_rate_data,
        }
        response_items.append(item)
        if last_checkin:
            hr_info = ""
            if heart_rate_data and heart_rate_data.get("avg_hr_bpm"):
                hr_info = f" | HR: {heart_rate_data['avg_hr_bpm']} bpm ({heart_rate_data['hr_quality']})"
            print(
                f"  {senior.get('firstName')} {senior.get('lastName')}: "
                f"{last_checkin.get('triage_status')} "
                f"(completed: {last_checkin.get('completed_at')}){hr_info}"
            )

    print(f"[DASHBOARD] Returning {len(response_items)} seniors")
    return {"seniors": response_items}
