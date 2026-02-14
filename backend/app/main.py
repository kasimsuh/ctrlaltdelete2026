from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
import ssl
import sys
import os
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pymongo

from app.db import mongo_check
from app.auth import (
    authenticate_user,
    create_access_token,
    create_user,
    ensure_user_indexes,
    require_current_user,
)

# Load env from repo root (or backend/) when running locally.
_backend_dir = Path(__file__).resolve().parent.parent  # backend/
load_dotenv(_backend_dir / ".env", override=False)
load_dotenv(_backend_dir.parent / ".env", override=False)

app = FastAPI(title="Guardian Check-In API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageStatus(str, Enum):
    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"


class CheckinStartRequest(BaseModel):
    demo_mode: bool = True
    senior_id: str = "demo-senior"


class CheckinStartResponse(BaseModel):
    checkin_id: str
    started_at: datetime


class Answers(BaseModel):
    dizziness: bool = False
    chest_pain: bool = False
    trouble_breathing: bool = False


class CheckinCompleteRequest(BaseModel):
    answers: Answers
    transcript: Optional[str] = None


class CheckinResult(BaseModel):
    checkin_id: str
    triage_status: TriageStatus
    triage_reasons: List[str]
    completed_at: datetime


class CheckinUploadResponse(BaseModel):
    checkin_id: str
    uploaded_at: datetime
    files: List[str]


class CheckinDetail(BaseModel):
    checkin_id: str
    senior_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    triage_status: Optional[TriageStatus] = None
    triage_reasons: List[str] = []
    transcript: Optional[str] = None


class CheckinListResponse(BaseModel):
    senior_id: str
    items: List[CheckinDetail]


class BaselineMetric(BaseModel):
    metric_type: str
    sample_count: int
    mean: float
    std_dev: float
    last_value: float
    updated_at: datetime


class BaselineResponse(BaseModel):
    senior_id: str
    metrics: List[BaselineMetric]


class WeeklySummaryRequest(BaseModel):
    week_start: Optional[str] = None


class WeeklySummary(BaseModel):
    summary_id: str
    senior_id: str
    week_start: str
    week_end: str
    summary_text: str
    key_trends: List[str]
    created_at: datetime


class WeeklySummaryResponse(BaseModel):
    senior_id: str
    summary: WeeklySummary


class AlertLevel(str, Enum):
    YELLOW = "yellow"
    RED = "red"


class AlertChannel(str, Enum):
    SMS = "sms"
    VOICE = "voice"
    EMAIL = "email"


class AlertRequest(BaseModel):
    senior_id: str = "demo-senior"
    level: AlertLevel
    channel: AlertChannel
    target: str
    message: Optional[str] = None


class AlertResponse(BaseModel):
    alert_id: str
    senior_id: str
    level: AlertLevel
    channel: AlertChannel
    target: str
    status: str
    created_at: datetime


class HealthStatus(BaseModel):
    status: str = Field(default="ok")
    time: datetime
    mongo: str = Field(default="unknown")
    mongo_host: Optional[str] = None
    mongo_db: Optional[str] = None
    mongo_error: Optional[str] = None
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    ssl: str = Field(default_factory=lambda: getattr(ssl, "OPENSSL_VERSION", "unknown"))
    pymongo: str = Field(default_factory=lambda: getattr(pymongo, "__version__", "unknown"))
    auth_required: bool = Field(default=False)
    jwt_configured: bool = Field(default=False)

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    email: str


CHECKINS: Dict[str, Dict[str, object]] = {}
CHECKIN_UPLOADS: Dict[str, List[str]] = {}
BASELINES: Dict[str, List[BaselineMetric]] = {}
ALERTS: Dict[str, List[AlertResponse]] = {}
WEEKLY_SUMMARIES: Dict[str, List[WeeklySummary]] = {}


@app.get("/health", response_model=HealthStatus)
def health_check() -> HealthStatus:
    ok, summary, err = mongo_check()
    return HealthStatus(
        time=datetime.utcnow(),
        mongo="ok" if ok else "error",
        mongo_host=summary.get("host"),
        mongo_db=summary.get("db"),
        mongo_error=err,
        auth_required=(os.environ.get("REQUIRE_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}),
        jwt_configured=bool(os.environ.get("JWT_SECRET")),
    )

@app.on_event("startup")
def _startup() -> None:
    # Ensure indexes when Mongo is reachable (Atlas/local).
    try:
        ensure_user_indexes()
    except Exception:
        pass


@app.post("/auth/register", response_model=MeResponse)
def register(payload: RegisterRequest) -> MeResponse:
    user = create_user(email=payload.email.strip().lower(), password=payload.password)
    return MeResponse(email=user["email"])


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = authenticate_user(email=payload.email.strip().lower(), password=payload.password)
    token = create_access_token(sub=str(user["_id"]), email=user["email"])
    return TokenResponse(access_token=token)


@app.get("/me", response_model=MeResponse)
def me(user: Optional[dict] = Depends(require_current_user)) -> MeResponse:
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return MeResponse(email=user["email"])


@app.post("/checkins/start", response_model=CheckinStartResponse)
def start_checkin(payload: CheckinStartRequest) -> CheckinStartResponse:
    checkin_id = str(uuid4())
    CHECKINS[checkin_id] = {
        "senior_id": payload.senior_id,
        "demo_mode": payload.demo_mode,
        "started_at": datetime.utcnow(),
        "status": "in_progress",
    }
    return CheckinStartResponse(checkin_id=checkin_id, started_at=CHECKINS[checkin_id]["started_at"])  # type: ignore[arg-type]


@app.post("/checkins/{checkin_id}/upload", response_model=CheckinUploadResponse)
def upload_checkin_artifacts(
    checkin_id: str,
    video: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    frames: Optional[List[UploadFile]] = File(default=None),
    metadata: Optional[str] = Form(default=None),
) -> CheckinUploadResponse:
    checkin = CHECKINS.get(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")

    files: List[str] = []
    if video is not None:
        files.append(video.filename)
    if audio is not None:
        files.append(audio.filename)
    if frames:
        files.extend([frame.filename for frame in frames])

    if metadata:
        files.append("metadata")

    CHECKIN_UPLOADS[checkin_id] = files
    return CheckinUploadResponse(
        checkin_id=checkin_id,
        uploaded_at=datetime.utcnow(),
        files=files,
    )


@app.post("/checkins/{checkin_id}/complete", response_model=CheckinResult)
def complete_checkin(checkin_id: str, payload: CheckinCompleteRequest) -> CheckinResult:
    checkin = CHECKINS.get(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")

    triage_status, triage_reasons = _triage(payload.answers)
    checkin.update(
        {
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "triage_status": triage_status,
            "triage_reasons": triage_reasons,
            "transcript": payload.transcript,
        }
    )

    return CheckinResult(
        checkin_id=checkin_id,
        triage_status=triage_status,
        triage_reasons=triage_reasons,
        completed_at=checkin["completed_at"],  # type: ignore[arg-type]
    )


@app.get("/checkins/{checkin_id}", response_model=CheckinDetail)
def get_checkin(checkin_id: str) -> CheckinDetail:
    checkin = CHECKINS.get(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")

    return CheckinDetail(
        checkin_id=checkin_id,
        senior_id=checkin.get("senior_id", "demo-senior"),
        status=checkin.get("status", "unknown"),
        started_at=checkin.get("started_at", datetime.utcnow()),
        completed_at=checkin.get("completed_at"),
        triage_status=checkin.get("triage_status"),
        triage_reasons=checkin.get("triage_reasons", []),
        transcript=checkin.get("transcript"),
    )


@app.get("/seniors/{senior_id}/checkins", response_model=CheckinListResponse)
def list_checkins(senior_id: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> CheckinListResponse:
    items: List[CheckinDetail] = []

    for checkin_id, checkin in CHECKINS.items():
        if checkin.get("senior_id", "demo-senior") != senior_id:
            continue

        items.append(
            CheckinDetail(
                checkin_id=checkin_id,
                senior_id=checkin.get("senior_id", "demo-senior"),
                status=checkin.get("status", "unknown"),
                started_at=checkin.get("started_at", datetime.utcnow()),
                completed_at=checkin.get("completed_at"),
                triage_status=checkin.get("triage_status"),
                triage_reasons=checkin.get("triage_reasons", []),
                transcript=checkin.get("transcript"),
            )
        )

    return CheckinListResponse(senior_id=senior_id, items=items)


@app.get("/seniors/{senior_id}/baseline", response_model=BaselineResponse)
def get_baseline(senior_id: str) -> BaselineResponse:
    metrics = BASELINES.get(senior_id, [])
    return BaselineResponse(senior_id=senior_id, metrics=metrics)


@app.post("/seniors/{senior_id}/summaries/weekly", response_model=WeeklySummaryResponse)
def create_weekly_summary(senior_id: str, payload: WeeklySummaryRequest) -> WeeklySummaryResponse:
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


@app.get("/seniors/{senior_id}/summaries/weekly", response_model=WeeklySummaryResponse)
def get_weekly_summary(senior_id: str, week_start: Optional[str] = None) -> WeeklySummaryResponse:
    summaries = WEEKLY_SUMMARIES.get(senior_id, [])
    if not summaries:
        raise HTTPException(status_code=404, detail="No weekly summaries available")

    if week_start:
        for summary in summaries:
            if summary.week_start == week_start:
                return WeeklySummaryResponse(senior_id=senior_id, summary=summary)
        raise HTTPException(status_code=404, detail="Weekly summary not found")

    return WeeklySummaryResponse(senior_id=senior_id, summary=summaries[-1])


@app.post("/alerts/test", response_model=AlertResponse)
def test_alert(payload: AlertRequest) -> AlertResponse:
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


@app.get("/seniors/{senior_id}/alerts", response_model=List[AlertResponse])
def list_alerts(senior_id: str) -> List[AlertResponse]:
    return ALERTS.get(senior_id, [])


def _triage(answers: Answers) -> tuple[TriageStatus, List[str]]:
    reasons: List[str] = []

    if answers.chest_pain or answers.trouble_breathing:
        reasons.append("Self-reported red flag symptom")
        return TriageStatus.RED, reasons

    if answers.dizziness:
        reasons.append("Reported dizziness")
        return TriageStatus.YELLOW, reasons

    reasons.append("No concerning signals detected")
    return TriageStatus.GREEN, reasons
