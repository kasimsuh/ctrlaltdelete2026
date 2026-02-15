from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
import json
import ssl
import sys
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pymongo

from app.auth import (
    require_current_user,
    ensure_user_indexes,
    create_user,
    authenticate_user,
    create_access_token,
)
from app.db import (
    mongo_check,
    get_dashboard_analytics,
    get_latest_checkins,
    get_senior_users,
    get_database,
)
from app.auth import hash_password
from bson import ObjectId

# Ensure backend/.env is loaded regardless of launch directory.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="Guardian Check-In API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageStatus(str, Enum):
    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"


class CheckinStartRequest(BaseModel):
    pass


class CheckinStartResponse(BaseModel):
    checkin_id: str
    started_at: datetime


class Answers(BaseModel):
    dizziness: bool = False
    chest_pain: bool = False
    trouble_breathing: bool = False
    medication_taken: Optional[bool] = None


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
    facial_symmetry: Optional["FacialSymmetryResult"] = None


class FacialSymmetrySummary(BaseModel):
    duration_s: float
    total_frames: int
    valid_frames: int
    quality_ratio: float
    symmetry_mean: Optional[float] = None
    symmetry_std: Optional[float] = None
    symmetry_p90: Optional[float] = None


class FacialSymmetryResult(BaseModel):
    status: str
    reason: str
    combined_index: Optional[float] = None
    rollups: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    summary: Optional[FacialSymmetrySummary] = None
    error: Optional[str] = None


class CheckinDetail(BaseModel):
    checkin_id: str
    senior_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    triage_status: Optional[TriageStatus] = None
    triage_reasons: List[str] = []
    transcript: Optional[str] = None
    facial_symmetry: Optional[FacialSymmetryResult] = None


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
    senior_id: str
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


class HealthStatus(BaseModel):
    status: str = Field(default="ok")
    time: datetime
    mongo: str = Field(default="unknown")
    mongo_host: Optional[str] = None
    mongo_db: Optional[str] = None
    mongo_error: Optional[str] = None
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    ssl: str = Field(default_factory=lambda: getattr(ssl, "OPENSSL_VERSION", "unknown"))
    pymongo: str = Field(
        default_factory=lambda: getattr(pymongo, "__version__", "unknown")
    )
    auth_required: bool = Field(default=False)
    jwt_configured: bool = Field(default=False)


class RegisterRequest(BaseModel):
    firstName: str
    lastName: str
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
    firstName: str = ""
    lastName: str = ""
    role: str = "senior"


class EphemeralTokenResponse(BaseModel):
    token: str
    expires_at: datetime


CheckinUploadResponse.model_rebuild()


CHECKINS: Dict[str, Dict[str, object]] = {}
CHECKIN_UPLOADS: Dict[str, List[str]] = {}
BASELINES: Dict[str, List[BaselineMetric]] = {}
ALERTS: Dict[str, List[AlertResponse]] = {}
WEEKLY_SUMMARIES: Dict[str, List[WeeklySummary]] = {}
SCREENINGS: Dict[str, ScreeningSession] = {}


def _db():
    return get_database()


def _users_collection():
    return _db()["users"]


def _checkins_collection():
    return _db()["checkin_history"]


def _screenings_collection():
    return _db()["screenings"]


def _parse_object_id(value: Optional[str]) -> Optional[ObjectId]:
    if not value:
        return None
    try:
        return ObjectId(value)
    except Exception:
        return None


def _parse_duration_ms(metadata: Optional[str]) -> int:
    if not metadata:
        return 10000
    try:
        payload = json.loads(metadata)
    except json.JSONDecodeError:
        return 10000
    duration_ms = payload.get("duration_ms")
    if duration_ms is None:
        return 10000
    try:
        return max(1000, min(30000, int(duration_ms)))
    except (TypeError, ValueError):
        return 10000


def _normalize_rollup_values(
    rollups: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for metric_name, metric_rollup in rollups.items():
        metric_data: Dict[str, Any] = {}
        for key, value in metric_rollup.items():
            if isinstance(value, (int, float)):
                metric_data[key] = float(value)
            else:
                metric_data[key] = value
        normalized[metric_name] = metric_data
    return normalized


def _facial_level_to_schema(level: Optional[str]) -> str:
    if not level:
        return "normal"
    lowered = level.lower()
    if lowered == "warning":
        return "warn"
    if lowered == "alert":
        return "alert"
    return lowered


def _facial_symmetry_metrics_payload(
    result: Optional[FacialSymmetryResult],
) -> Optional[Dict[str, Any]]:
    if result is None:
        return None

    rollups = result.rollups or {}
    summary = result.summary
    combined_raw = result.combined_index or 0.0
    combined_index = min(1.0, max(0.0, float(combined_raw) / 100.0))

    def rollup_payload(name: str) -> Dict[str, Any]:
        roll = rollups.get(name, {})
        return {
            "median_percent": float(roll.get("median", 0.0)),
            "p90_percent": float(roll.get("p90", 0.0)),
            "level": _facial_level_to_schema(str(roll.get("level", "normal"))),
        }

    payload = {
        "mouth": rollup_payload("mouth"),
        "eye": rollup_payload("eye"),
        "nasolabial": rollup_payload("nasolabial"),
        "combined_index": combined_index,
        "quality": {
            "valid_frames": int(summary.valid_frames) if summary else 0,
            "total_frames": int(summary.total_frames) if summary else 0,
            "quality_ratio": float(summary.quality_ratio) if summary else 0.0,
            "duration_seconds": float(summary.duration_s) if summary else 0.0,
            "index_mean": float(summary.symmetry_mean)
            if summary and summary.symmetry_mean is not None
            else None,
            "index_std": float(summary.symmetry_std)
            if summary and summary.symmetry_std is not None
            else None,
        },
    }

    return payload


def _merge_triage(
    answers: Answers,
    facial_symmetry: Optional[FacialSymmetryResult],
) -> Tuple[TriageStatus, List[str]]:
    status, reasons = _triage(answers)

    if facial_symmetry is None:
        return status, reasons

    facial_status = (facial_symmetry.status or "").upper()
    if facial_status in {"RED", "YELLOW", "GREEN"}:
        reasons.append(f"Facial symmetry: {facial_symmetry.reason}")
    elif facial_status in {"ERROR", "RETRY"}:
        reasons.append("Facial symmetry check needs retry.")

    if status == TriageStatus.RED:
        return status, reasons

    if facial_status == "RED":
        return TriageStatus.RED, reasons
    if facial_status in {"YELLOW", "ERROR", "RETRY"} and status == TriageStatus.GREEN:
        return TriageStatus.YELLOW, reasons

    return status, reasons


def _run_facial_symmetry(video_bytes: bytes, duration_ms: int) -> FacialSymmetryResult:
    if not video_bytes:
        return FacialSymmetryResult(
            status="ERROR",
            reason="Uploaded camera clip is empty.",
            error="empty_video",
        )

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    try:
        import cv2  # type: ignore
        from run_facial_symmetry_checkin import classify_scientific_index
        from vision.facial_symmetry import FacialSymmetryAnalyzer, summarize_session
    except Exception as exc:
        return FacialSymmetryResult(
            status="ERROR",
            reason="Facial symmetry dependencies are not available.",
            error=f"{exc.__class__.__name__}: {exc}",
        )

    clip_seconds = duration_ms / 1000.0
    fallback_fps = 30.0
    cap = None
    analyzer = None
    temp_path = None
    samples = []
    frame_count = 0

    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return FacialSymmetryResult(
                status="ERROR",
                reason="Uploaded camera clip could not be decoded.",
                error="video_decode_failed",
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        has_valid_fps = fps >= 1.0
        analyzer = FacialSymmetryAnalyzer(model_path=None)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            elapsed_s = (
                (frame_count / fps) if has_valid_fps else (frame_count / fallback_fps)
            )
            if elapsed_s >= clip_seconds:
                break

            metrics = analyzer.process_frame(frame, elapsed_s)
            if metrics is not None:
                samples.append(metrics)
            frame_count += 1

        analyzed_duration_s = min(
            clip_seconds,
            (frame_count / fps) if has_valid_fps else (frame_count / fallback_fps),
        )
        summary = summarize_session(samples, duration_s=analyzed_duration_s)
        triage, rollups, combined_index = classify_scientific_index(
            samples, sensitivity=1.0
        )
        return FacialSymmetryResult(
            status=triage["status"],
            reason=triage["reason"],
            combined_index=float(combined_index),
            rollups=_normalize_rollup_values(rollups),
            summary=FacialSymmetrySummary(
                duration_s=float(summary.duration_s),
                total_frames=int(summary.total_frames),
                valid_frames=int(summary.valid_frames),
                quality_ratio=float(summary.quality_ratio),
                symmetry_mean=(
                    float(summary.symmetry_mean)
                    if summary.symmetry_mean is not None
                    else None
                ),
                symmetry_std=(
                    float(summary.symmetry_std)
                    if summary.symmetry_std is not None
                    else None
                ),
                symmetry_p90=(
                    float(summary.symmetry_p90)
                    if summary.symmetry_p90 is not None
                    else None
                ),
            ),
        )
    except Exception as exc:
        return FacialSymmetryResult(
            status="ERROR",
            reason="Facial symmetry analysis failed.",
            error=f"{exc.__class__.__name__}: {exc}",
        )
    finally:
        if analyzer is not None:
            analyzer.close()
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.get("/health", response_model=HealthStatus)
def health_check() -> HealthStatus:
    ok, summary, err = mongo_check()
    return HealthStatus(
        time=datetime.utcnow(),
        mongo="ok" if ok else "error",
        mongo_host=summary.get("host"),
        mongo_db=summary.get("db"),
        mongo_error=err,
        auth_required=(
            os.environ.get("REQUIRE_AUTH", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
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
    user = create_user(
        email=payload.email.strip().lower(),
        password=payload.password,
        firstName=payload.firstName,
        lastName=payload.lastName,
    )
    return MeResponse(
        email=user["email"],
        firstName=user.get("firstName", ""),
        lastName=user.get("lastName", ""),
        role=user.get("role", "senior"),
    )


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = authenticate_user(
        email=payload.email.strip().lower(), password=payload.password
    )
    token = create_access_token(sub=str(user["_id"]), email=user["email"])
    return TokenResponse(access_token=token)


@app.get("/me", response_model=MeResponse)
def me(user: Optional[dict] = Depends(require_current_user)) -> MeResponse:
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return MeResponse(
        email=user["email"],
        firstName=user.get("firstName", ""),
        lastName=user.get("lastName", ""),
        role=user.get("role", "senior"),
    )


def _require_doctor(user: Optional[dict]) -> dict:
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctor role required")
    return user


def _serialize_dt(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _screening_transcript(responses: List[ScreeningResponseItem]) -> str:
    lines: List[str] = []
    for item in responses:
        if item.q:
            lines.append(f"AI: {item.q}")
        if item.transcript:
            lines.append(f"USER: {item.transcript}")
    return " ".join(lines)


def _triage_status_from_db(value: Optional[str]) -> Optional[TriageStatus]:
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


def _load_checkin(checkin_id: str) -> Optional[Dict[str, object]]:
    checkin = CHECKINS.get(checkin_id)
    if checkin is not None:
        return checkin

    doc = _checkins_collection().find_one({"checkin_id": checkin_id})
    if not doc:
        return None

    checkin = {
        "senior_id": str(doc.get("user_id", "")),
        "demo_mode": False,
        "started_at": doc.get("started_at"),
        "status": doc.get("status", "unknown"),
        "completed_at": doc.get("completed_at"),
        "triage_status": _triage_status_from_db(doc.get("triage_status")),
        "triage_reasons": doc.get("triage_reasons", []),
        "transcript": doc.get("transcript"),
        "user_id": doc.get("user_id"),
    }

    if doc.get("facial_symmetry_raw"):
        checkin["facial_symmetry"] = doc.get("facial_symmetry_raw")

    CHECKINS[checkin_id] = checkin
    return checkin


@app.get("/dashboard/analytics")
def dashboard_analytics(user: Optional[dict] = Depends(require_current_user)):
    _require_doctor(user)
    analytics = get_dashboard_analytics(days=7)
    print(f"[DASHBOARD] Analytics loaded for doctor {user.get('email')}: {analytics}")
    return analytics


@app.get("/dashboard/seniors")
def dashboard_seniors(user: Optional[dict] = Depends(require_current_user)):
    _require_doctor(user)
    seniors = get_senior_users(limit=50)
    user_ids = [senior["_id"] for senior in seniors]
    latest_checkins = get_latest_checkins(user_ids)

    print(f"[DASHBOARD] Fetching seniors list for doctor {user.get('email')}")
    print(f"[DASHBOARD] Total seniors: {len(seniors)}")

    response_items = []
    for senior in seniors:
        last_checkin = latest_checkins.get(senior["_id"])
        item = {
            "id": str(senior["_id"]),
            "firstName": senior.get("firstName", ""),
            "lastName": senior.get("lastName", ""),
            "email": senior.get("email", ""),
            "lastCheckinAt": _serialize_dt(last_checkin.get("completed_at")) if last_checkin else None,
            "triageStatus": (last_checkin.get("triage_status") if last_checkin else None),
            "checkinId": (last_checkin.get("checkin_id") if last_checkin else None),
        }
        response_items.append(item)
        if last_checkin:
            print(f"  {senior.get('firstName')} {senior.get('lastName')}: {last_checkin.get('triage_status')} (completed: {last_checkin.get('completed_at')})")

    print(f"[DASHBOARD] Returning {len(response_items)} seniors")
    return {"seniors": response_items}


@app.post("/auth/ephemeral", response_model=EphemeralTokenResponse)
@app.get("/auth/ephemeral", response_model=EphemeralTokenResponse)
def create_ephemeral_token() -> EphemeralTokenResponse:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set (set it in your shell env or in the repo root .env)",
        )

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"},
    )

    now = datetime.now(timezone.utc)
    try:
        token = client.auth_tokens.create(
            config={
                "uses": 1,
                "expire_time": now + timedelta(minutes=30),
                "new_session_expire_time": now + timedelta(minutes=5),
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to create Gemini ephemeral token: {exc.__class__.__name__}",
        ) from exc

    return EphemeralTokenResponse(
        token=token.name, expires_at=now + timedelta(minutes=30)
    )


@app.post("/checkins/start", response_model=CheckinStartResponse)
def start_checkin(
    payload: CheckinStartRequest,
    user: Optional[dict] = Depends(require_current_user),
) -> CheckinStartResponse:
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required to start check-in")
    
    checkin_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    senior_user = user

    checkin_doc = {
        "user_id": senior_user["_id"],
        "checkin_id": checkin_id,
        "started_at": started_at,
        "status": "in_progress",
        "created_at": started_at,
        "completed_at": None,
        "triage_status": None,
        "triage_reasons": [],
        "answers": {},
        "transcript": None,
        "screening_session_id": None,
        "screening_responses": [],
        "metrics": {},
        "facial_symmetry_raw": None,
        "user_message": None,
        "clinician_notes": None,
        "alert_level": None,
        "alert_sent": False,
        "alert_target": None,
        "alert_message": None,
        "alert_sent_at": None,
    }
    _checkins_collection().insert_one(checkin_doc)

    CHECKINS[checkin_id] = {
        "senior_id": str(senior_user["_id"]),
        "demo_mode": False,
        "started_at": started_at,
        "status": "in_progress",
        "user_id": senior_user["_id"],
    }
    return CheckinStartResponse(checkin_id=checkin_id, started_at=started_at)


@app.post("/checkins/{checkin_id}/upload", response_model=CheckinUploadResponse)
def upload_checkin_artifacts(
    checkin_id: str,
    video: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    frames: Optional[List[UploadFile]] = File(default=None),
    metadata: Optional[str] = Form(default=None),
) -> CheckinUploadResponse:
    checkin = _load_checkin(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")

    files: List[str] = []
    facial_symmetry: Optional[FacialSymmetryResult] = None
    duration_ms = _parse_duration_ms(metadata)

    if video is not None:
        files.append(video.filename or "video")
        video_bytes = video.file.read()
        facial_symmetry = _run_facial_symmetry(video_bytes, duration_ms)
        checkin["facial_symmetry"] = facial_symmetry.model_dump()
    if audio is not None:
        files.append(audio.filename or "audio")
    if frames:
        files.extend([frame.filename or "frame" for frame in frames])

    if metadata:
        files.append("metadata")

    CHECKIN_UPLOADS[checkin_id] = files

    if facial_symmetry is not None:
        metrics_payload = _facial_symmetry_metrics_payload(facial_symmetry)
        _checkins_collection().update_one(
            {"checkin_id": checkin_id},
            {
                "$set": {
                    "metrics.facial_symmetry": metrics_payload,
                    "facial_symmetry_raw": facial_symmetry.model_dump(),
                }
            },
        )
    return CheckinUploadResponse(
        checkin_id=checkin_id,
        uploaded_at=datetime.utcnow(),
        files=files,
        facial_symmetry=facial_symmetry,
    )


@app.post("/checkins/{checkin_id}/complete", response_model=CheckinResult)
def complete_checkin(checkin_id: str, payload: CheckinCompleteRequest) -> CheckinResult:
    checkin = _load_checkin(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")

    facial_symmetry = None
    if checkin.get("facial_symmetry"):
        facial_symmetry = FacialSymmetryResult(**checkin["facial_symmetry"])

    triage_status, triage_reasons = _merge_triage(payload.answers, facial_symmetry)
    checkin.update(
        {
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "triage_status": triage_status,
            "triage_reasons": triage_reasons,
            "transcript": payload.transcript,
        }
    )

    screening = None
    try:
        screening = _screenings_collection().find_one({"checkin_id": checkin_id})
    except Exception:
        screening = None

    screening_responses = screening.get("responses") if screening else None
    screening_session_id = screening.get("session_id") if screening else None
    transcript = payload.transcript or (screening.get("transcript") if screening else None)
    if transcript is None and screening_responses:
        transcript = _screening_transcript(
            [ScreeningResponseItem(**item) for item in screening_responses]
        )

    triage_status_db = triage_status.value.lower()

    _checkins_collection().update_one(
        {"checkin_id": checkin_id},
        {
            "$set": {
                "status": "completed",
                "completed_at": checkin["completed_at"],
                "triage_status": triage_status_db,
                "triage_reasons": triage_reasons,
                "answers": payload.answers.model_dump(),
                "screening_session_id": screening_session_id or None,
                "screening_responses": screening_responses or [],
                "transcript": transcript or None,
                "user_message": "Check-in completed.",
                "clinician_notes": "",
                "alert_level": None,
                "alert_sent": False,
                "alert_target": None,
                "alert_message": None,
                "alert_sent_at": None,
            }
        },
    )

    return CheckinResult(
        checkin_id=checkin_id,
        triage_status=triage_status,
        triage_reasons=triage_reasons,
        completed_at=checkin["completed_at"],  # type: ignore[arg-type]
    )


@app.get("/checkins/{checkin_id}", response_model=CheckinDetail)
def get_checkin(checkin_id: str) -> CheckinDetail:
    checkin = _load_checkin(checkin_id)
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
        facial_symmetry=checkin.get("facial_symmetry"),
    )


@app.get("/seniors/{senior_id}/checkins", response_model=CheckinListResponse)
def list_checkins(
    senior_id: str, from_date: Optional[str] = None, to_date: Optional[str] = None
) -> CheckinListResponse:
    items: List[CheckinDetail] = []

    # Parse senior_id as ObjectId
    try:
        user_id = ObjectId(senior_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid senior_id format")

    query: Dict[str, Any] = {"user_id": user_id}
    if from_date or to_date:
        date_filter: Dict[str, Any] = {}
        if from_date:
            date_filter["$gte"] = datetime.fromisoformat(from_date)
        if to_date:
            date_filter["$lte"] = datetime.fromisoformat(to_date)
        query["completed_at"] = date_filter

    docs = _checkins_collection().find(query).sort("completed_at", -1)
    for doc in docs:
        facial_symmetry_raw = doc.get("facial_symmetry_raw")
        items.append(
            CheckinDetail(
                checkin_id=doc.get("checkin_id", ""),
                senior_id=str(doc.get("user_id", "")),
                status=doc.get("status", "unknown"),
                started_at=doc.get("started_at", datetime.utcnow()),
                completed_at=doc.get("completed_at"),
                triage_status=_triage_status_from_db(doc.get("triage_status")),
                triage_reasons=doc.get("triage_reasons", []),
                transcript=doc.get("transcript"),
                facial_symmetry=facial_symmetry_raw,
            )
        )

    return CheckinListResponse(senior_id=senior_id, items=items)


@app.get("/seniors/{senior_id}/baseline", response_model=BaselineResponse)
def get_baseline(senior_id: str) -> BaselineResponse:
    metrics = BASELINES.get(senior_id, [])
    return BaselineResponse(senior_id=senior_id, metrics=metrics)


@app.post("/seniors/{senior_id}/summaries/weekly", response_model=WeeklySummaryResponse)
def create_weekly_summary(
    senior_id: str, payload: WeeklySummaryRequest
) -> WeeklySummaryResponse:
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
def get_weekly_summary(
    senior_id: str, week_start: Optional[str] = None
) -> WeeklySummaryResponse:
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


@app.post("/screenings", response_model=ScreeningCreateResponse)
def create_screening(payload: ScreeningCreateRequest) -> ScreeningCreateResponse:
    if not payload.checkin_id:
        raise HTTPException(status_code=400, detail="checkin_id is required")
    
    session_id = payload.session_id or f"screening-{uuid4()}"
    timestamp = payload.timestamp or datetime.utcnow()
    
    # Get checkin to find the user
    checkin_doc = _checkins_collection().find_one({"checkin_id": payload.checkin_id})
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

    _screenings_collection().insert_one(
        {
            "session_id": session_id,
            "senior_id": senior_id,
            "checkin_id": payload.checkin_id,
            "timestamp": timestamp,
            "responses": [item.model_dump() for item in payload.responses],
            "transcript": _screening_transcript(payload.responses),
        }
    )

    if payload.checkin_id:
        _checkins_collection().update_one(
            {"checkin_id": payload.checkin_id},
            {
                "$set": {
                    "screening_session_id": session_id,
                    "screening_responses": [item.model_dump() for item in payload.responses],
                    "transcript": _screening_transcript(payload.responses),
                }
            },
        )
    return ScreeningCreateResponse(session_id=session_id, stored_at=datetime.utcnow())


@app.get("/screenings/{session_id}", response_model=ScreeningSession)
def get_screening(session_id: str) -> ScreeningSession:
    session = SCREENINGS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Screening session not found")
    return session


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
