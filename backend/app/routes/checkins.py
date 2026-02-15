"""Check-in management routes."""

import base64
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth import require_current_user
from app.dependencies import (
    get_checkins_collection,
    get_screenings_collection,
    get_users_collection,
)
from app.models.checkin import (
    Answers,
    CheckinStartRequest,
    CheckinStartResponse,
    CheckinCompleteRequest,
    CheckinResult,
    CheckinUploadResponse,
    CheckinDetail,
    FacialSymmetryResult,
    HeartRateResult,
    TriageStatus,
)
from app.models.screening import ScreeningResponseItem
from app.services.triage import merge_triage
from app.services.facial_symmetry import (
    run_facial_symmetry_analysis,
    build_facial_symmetry_metrics,
)
from app.services.screening import build_screening_transcript
from app.services.stt_assessment import assess_checkin_with_stt
from app.vhr import analyze_uploaded_video

router = APIRouter()
logger = logging.getLogger("guardian")

# In-memory cache for active check-ins
CHECKINS: Dict[str, Dict[str, object]] = {}
CHECKIN_UPLOADS: Dict[str, List[str]] = {}


class CheckinValidationError:
    """Represents a validation issue with a check-in."""

    def __init__(self, field: str, message: str, is_blocking: bool = True):
        self.field = field
        self.message = message
        self.is_blocking = is_blocking


def _validate_checkin_completeness(
    checkin: Dict[str, object],
    answers: Answers,
    facial_symmetry: Optional[FacialSymmetryResult],
    transcript: Optional[str],
    screening_responses: Optional[List[dict]],
) -> List[CheckinValidationError]:
    """
    Validate that a check-in has all required data before completion.
    Returns a list of validation errors (empty if valid).
    """
    errors: List[CheckinValidationError] = []

    # Check facial symmetry status
    if facial_symmetry is not None:
        status = (facial_symmetry.status or "").upper()
        if status in {"ERROR", "RETRY"}:
            errors.append(
                CheckinValidationError(
                    field="facial_symmetry",
                    message=f"Facial symmetry analysis failed: {facial_symmetry.reason or 'needs retry'}",
                    is_blocking=True,
                )
            )
    else:
        # No facial symmetry data at all - this might be acceptable in some cases
        errors.append(
            CheckinValidationError(
                field="facial_symmetry",
                message="No facial symmetry data recorded",
                is_blocking=False,  # Warning, not blocking
            )
        )

    # Check if we have any screening data
    has_transcript = bool(transcript and transcript.strip())
    has_screening = bool(screening_responses and len(screening_responses) > 0)

    if not has_transcript and not has_screening:
        errors.append(
            CheckinValidationError(
                field="screening",
                message="No screening transcript or responses recorded",
                is_blocking=True,
            )
        )

    # Check if all answers are false/none (no actual input from user)
    answers_dict = answers.model_dump()
    all_answers_empty = all(
        v is None or v is False for v in answers_dict.values()
    )
    if all_answers_empty and not has_transcript:
        errors.append(
            CheckinValidationError(
                field="answers",
                message="No symptoms or responses recorded from user",
                is_blocking=True,
            )
        )

    return errors


def _parse_duration_ms(metadata: Optional[str]) -> int:
    """Parse duration from metadata JSON."""
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


def _triage_reasons_from_stt_items(items: list[dict]) -> list[str]:
    """
    Best-effort extraction of triage reasons from backend/logs/stt.json.
    stt.json is overwritten, so this is only used when email matches the check-in user.
    """

    def _ans(idx: int) -> str:
        try:
            return str(items[idx].get("answer") or "").strip().lower()
        except Exception:
            return ""

    reasons: list[str] = []

    # Q2 is expected to be symptoms.
    symptoms = _ans(1)
    if "dizz" in symptoms:
        reasons.append("Reported dizziness (speech)")
    if "chest" in symptoms:
        reasons.append("Reported chest pain (speech)")
    if "breath" in symptoms or "shortness" in symptoms:
        reasons.append("Reported trouble breathing (speech)")

    # Q3 is expected to be medications.
    meds = _ans(2)
    if meds and any(tok in meds for tok in ("no", "not", "forgot", "didn't", "didnt")):
        reasons.append("Medication not taken (speech)")

    return reasons


def _maybe_merge_stt_reasons(checkin: dict, triage_reasons: list[str]) -> list[str]:
    """
    Merge derived triage reasons from stt.json into triage_reasons.
    """
    try:
        user_id = checkin.get("user_id")
        if not user_id:
            return triage_reasons

        stt_path = Path(__file__).resolve().parents[2] / "logs" / "stt.json"
        if not stt_path.exists():
            return triage_reasons
        stt_payload = json.loads(stt_path.read_text(encoding="utf-8"))
        stt_user_id = str(stt_payload.get("user_id") or "").strip()
        if not stt_user_id or stt_user_id != str(user_id):
            return triage_reasons

        items = stt_payload.get("items") or []
        if not isinstance(items, list):
            return triage_reasons

        derived = _triage_reasons_from_stt_items(items)
        if not derived:
            return triage_reasons

        existing = {str(r).strip().lower() for r in triage_reasons if str(r).strip()}
        merged = list(triage_reasons)
        for r in derived:
            if r.strip().lower() not in existing:
                merged.append(r)
        return merged
    except Exception:
        return triage_reasons


def _load_stt_items_for_checkin_user(checkin: dict) -> list[dict]:
    """
    Load backend/logs/stt.json if it belongs to the user who created this check-in.
    The file is overwritten each run; this check prevents cross-user leakage.
    """
    try:
        user_id = checkin.get("user_id")
        if not user_id:
            return []

        stt_path = Path(__file__).resolve().parents[2] / "logs" / "stt.json"
        if not stt_path.exists():
            return []

        stt_payload = json.loads(stt_path.read_text(encoding="utf-8"))
        stt_user_id = str(stt_payload.get("user_id") or "").strip()
        if stt_user_id and stt_user_id != str(user_id):
            return []
        # Backward-compat: older stt.json used email; keep this as a fallback.
        if not stt_user_id and stt_payload.get("email"):
            user = get_users_collection().find_one({"_id": user_id})
            user_email = str((user or {}).get("email") or "").strip().lower()
            stt_email = str(stt_payload.get("email") or "").strip().lower()
            if not user_email or stt_email != user_email:
                return []

        items = stt_payload.get("items") or []
        if not isinstance(items, list):
            return []
        return [i for i in items if isinstance(i, dict)]
    except Exception:
        return []


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


def _triage_from_ai_signals(
    signals: Optional[dict],
    facial_symmetry: Optional[FacialSymmetryResult],
) -> Optional[TriageStatus]:
    """
    Compute triage from Gemini's negation-aware signals.
    Returns None when signals are missing/invalid.
    """
    if not isinstance(signals, dict) or not signals:
        return None

    def _facial_severity(result: Optional[FacialSymmetryResult]) -> Optional[TriageStatus]:
        if result is None:
            return None

        status = (result.status or "").upper()
        if status == "RED":
            return TriageStatus.RED
        if status == "YELLOW":
            return TriageStatus.YELLOW
        if status == "GREEN":
            return TriageStatus.GREEN
        if status in {"ERROR", "RETRY"}:
            # Conservative: if facial analysis failed, do not mark GREEN.
            return TriageStatus.YELLOW
        if status == "SKIPPED":
            return None

        # Fallback when status is unknown but numeric index exists.
        # combined_index from the facial pipeline is typically 0..100.
        quality = None
        if result.summary is not None:
            try:
                quality = float(result.summary.quality_ratio)
            except Exception:
                quality = None
        if quality is not None and quality < 0.5:
            return None

        if result.combined_index is None:
            return None
        try:
            idx = max(0.0, min(1.0, float(result.combined_index) / 100.0))
        except Exception:
            return None

        if idx >= 0.85:
            return TriageStatus.RED
        if idx >= 0.65:
            return TriageStatus.YELLOW
        return TriageStatus.GREEN

    chest = signals.get("chest_pain")
    breath = signals.get("trouble_breathing")
    dizzy = signals.get("dizziness")

    speech = TriageStatus.GREEN
    if chest == "present" or breath == "present":
        speech = TriageStatus.RED
    elif dizzy == "present":
        speech = TriageStatus.YELLOW

    facial = _facial_severity(facial_symmetry)

    # Combine: take the more severe of (speech, facial).
    if speech == TriageStatus.RED or facial == TriageStatus.RED:
        return TriageStatus.RED
    if speech == TriageStatus.YELLOW or facial == TriageStatus.YELLOW:
        return TriageStatus.YELLOW
    return TriageStatus.GREEN


def _load_checkin(checkin_id: str) -> Optional[Dict[str, object]]:
    """Load check-in from cache or database."""
    checkin = CHECKINS.get(checkin_id)
    if checkin is not None:
        return checkin

    doc = get_checkins_collection().find_one({"checkin_id": checkin_id})
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
        "ai_assessment": doc.get("ai_assessment"),
        "camera_snapshot": doc.get("camera_snapshot"),
    }

    if doc.get("facial_symmetry_raw"):
        checkin["facial_symmetry"] = doc.get("facial_symmetry_raw")
    
    if doc.get("heart_rate_raw"):
        checkin["heart_rate"] = doc.get("heart_rate_raw")

    CHECKINS[checkin_id] = checkin
    return checkin


def _auto_complete_abandoned_checkin(checkin_id: str) -> bool:
    """
    Auto-complete an abandoned check-in if it has screening data.
    Returns True if completed, False otherwise.
    """
    doc = get_checkins_collection().find_one({"checkin_id": checkin_id})
    if not doc:
        return False
    
    # Only auto-complete if still in_progress and has screening data
    if doc.get("status") != "in_progress":
        return False
    
    transcript = doc.get("transcript")
    screening_responses = doc.get("screening_responses", [])
    facial_symmetry_raw = doc.get("facial_symmetry_raw")
    
    # Need at least screening data to auto-complete
    if not transcript and not screening_responses:
        return False
    
    # Extract answers from transcript or use defaults
    answers = Answers(
        dizziness="dizziness" in (transcript or "").lower(),
        chest_pain="chest pain" in (transcript or "").lower(),
        trouble_breathing="trouble breathing" in (transcript or "").lower(),
        medication_taken=None,
    )
    
    facial_symmetry = None
    if facial_symmetry_raw:
        facial_symmetry = FacialSymmetryResult(**facial_symmetry_raw)
    
    triage_status, triage_reasons = merge_triage(answers, facial_symmetry)
    completed_at = datetime.utcnow()
    
    get_checkins_collection().update_one(
        {"checkin_id": checkin_id},
        {
            "$set": {
                "status": "completed",
                "completed_at": completed_at,
                "triage_status": triage_status.value.lower(),
                "triage_reasons": triage_reasons,
                "answers": answers.model_dump(),
                "user_message": "Check-in auto-completed from screening data.",
            }
        },
    )
    
    return True


@router.post("/start", response_model=CheckinStartResponse)
def start_checkin(
    payload: CheckinStartRequest,
    user: Optional[dict] = Depends(require_current_user),
) -> CheckinStartResponse:
    """Start a new check-in session."""
    if user is None:
        raise HTTPException(
            status_code=401, detail="Authentication required to start check-in"
        )

    checkin_id = str(uuid4())
    started_at = datetime.now(timezone.utc)

    checkin_doc = {
        "user_id": user["_id"],
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
        "heart_rate_raw": None,
        "camera_snapshot": None,
        "user_message": None,
        "clinician_notes": None,
        "alert_level": None,
        "alert_sent": False,
        "alert_target": None,
        "alert_message": None,
        "alert_sent_at": None,
    }
    get_checkins_collection().insert_one(checkin_doc)

    CHECKINS[checkin_id] = {
        "senior_id": str(user["_id"]),
        "demo_mode": False,
        "started_at": started_at,
        "status": "in_progress",
        "user_id": user["_id"],
    }
    return CheckinStartResponse(checkin_id=checkin_id, started_at=started_at)


@router.post("/{checkin_id}/upload", response_model=CheckinUploadResponse)
async def upload_checkin_artifacts(
    checkin_id: str,
    video: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    frames: Optional[List[UploadFile]] = File(default=None),
    metadata: Optional[str] = Form(default=None),
) -> CheckinUploadResponse:
    """Upload video/audio artifacts for a check-in, analyze facial symmetry and heart rate."""
    checkin = _load_checkin(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")

    files: List[str] = []
    facial_symmetry: Optional[FacialSymmetryResult] = None
    heart_rate: Optional[HeartRateResult] = None
    duration_ms = _parse_duration_ms(metadata)

    if video is not None:
        files.append(video.filename or "video")
        
        # Read video once for both analyses
        video_bytes = await video.read()
        
        # Run facial symmetry analysis
        facial_symmetry = run_facial_symmetry_analysis(video_bytes, duration_ms)
        checkin["facial_symmetry"] = facial_symmetry.model_dump()
        
        # Run VHR analysis (requires re-uploading since it's async)
        # Reset video position for VHR analysis
        await video.seek(0)
        try:
            vhr_result = await analyze_uploaded_video(video)
            heart_rate = HeartRateResult(**vhr_result)
            checkin["heart_rate"] = heart_rate.model_dump()
        except Exception as e:
            # VHR analysis is optional, don't fail the entire upload
            heart_rate = HeartRateResult(
                avg_hr_bpm=None,
                hr_quality="low",
                note=f"VHR analysis failed: {str(e)}",
            )
            checkin["heart_rate"] = heart_rate.model_dump()
    
    if audio is not None:
        files.append(audio.filename or "audio")
    if frames:
        files.extend([frame.filename or "frame" for frame in frames])

    if metadata:
        files.append("metadata")

    CHECKIN_UPLOADS[checkin_id] = files

    # Update database with facial symmetry, heart rate, and an optional snapshot frame.
    update_doc = {}
    if facial_symmetry is not None:
        metrics_payload = build_facial_symmetry_metrics(facial_symmetry)
        update_doc["metrics.facial_symmetry"] = metrics_payload
        update_doc["facial_symmetry_raw"] = facial_symmetry.model_dump()
    
    if heart_rate is not None:
        update_doc["heart_rate_raw"] = heart_rate.model_dump()
        update_doc["metrics.heart_rate"] = {
            "avg_hr_bpm": heart_rate.avg_hr_bpm,
            "hr_quality": heart_rate.hr_quality,
            "sqi": heart_rate.sqi,
        }

    if frames:
        try:
            first = frames[0]
            frame_bytes = await first.read()
            # Keep payload small; a 360px JPEG should fit easily.
            if frame_bytes and len(frame_bytes) <= 450_000:
                data_url = "data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode("ascii")
                update_doc["camera_snapshot"] = data_url
                checkin["camera_snapshot"] = data_url
        except Exception:
            # Snapshot is optional; don't fail the upload.
            pass
    
    if update_doc:
        get_checkins_collection().update_one(
            {"checkin_id": checkin_id},
            {"$set": update_doc},
        )
    
    return CheckinUploadResponse(
        checkin_id=checkin_id,
        uploaded_at=datetime.utcnow(),
        files=files,
        facial_symmetry=facial_symmetry,
        heart_rate=heart_rate,
    )


@router.post("/{checkin_id}/complete", response_model=CheckinResult)
def complete_checkin(
    checkin_id: str,
    payload: CheckinCompleteRequest,
    force: bool = False,
) -> CheckinResult:
    """
    Complete a check-in with answers and triage determination.

    Args:
        checkin_id: The check-in to complete
        payload: Answers and transcript from the screening
        force: If True, allows completion even with non-blocking validation warnings

    Raises:
        HTTPException 422: If required data is missing (facial symmetry failed, no screening data)
    """
    checkin = _load_checkin(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")

    facial_symmetry = None
    if checkin.get("facial_symmetry"):
        facial_symmetry = FacialSymmetryResult(**checkin["facial_symmetry"])

    triage_status, triage_reasons = merge_triage(payload.answers, facial_symmetry)
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
        screening = get_screenings_collection().find_one({"checkin_id": checkin_id})
    except Exception:
        screening = None

    screening_responses = screening.get("responses") if screening else None
    screening_session_id = screening.get("session_id") if screening else None
    transcript = payload.transcript or (
        screening.get("transcript") if screening else None
    )
    if transcript is None and screening_responses:
        transcript = build_screening_transcript(
            [ScreeningResponseItem(**item) for item in screening_responses]
        )

    # Validate check-in completeness before saving
    validation_errors = _validate_checkin_completeness(
        checkin=checkin,
        answers=payload.answers,
        facial_symmetry=facial_symmetry,
        transcript=transcript,
        screening_responses=screening_responses,
    )

    # Check for blocking errors
    blocking_errors = [e for e in validation_errors if e.is_blocking]
    if blocking_errors and not force:
        error_messages = [f"{e.field}: {e.message}" for e in blocking_errors]
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Check-in data is incomplete",
                "errors": error_messages,
                "can_retry": True,
            },
        )
    if blocking_errors and force:
        # Allow completion, but preserve a note in the triage reasons.
        triage_reasons = list(triage_reasons) + [
            f"Validation warning (forced complete): {e.field}: {e.message}"
            for e in blocking_errors
        ]

    # Gemini assessment at check-in time (preferred over substring-based heuristics).
    stt_items = _load_stt_items_for_checkin_user(checkin)
    ai_assessment = assess_checkin_with_stt(
        stt_items=stt_items,
        answers=payload.answers.model_dump(),
        transcript=transcript,
        facial_symmetry=facial_symmetry.model_dump() if facial_symmetry else None,
        heart_rate=checkin.get("heart_rate") if isinstance(checkin.get("heart_rate"), dict) else None,
    )
    checkin["ai_assessment"] = ai_assessment
    try:
        logger.info(
            "checkin_ai_assessment generated checkin_id=%s model=%s has_stt=%s",
            checkin_id,
            (ai_assessment or {}).get("model") if isinstance(ai_assessment, dict) else None,
            bool(stt_items),
        )
    except Exception:
        pass

    # Prefer Gemini's negation-aware signals for triage when available.
    try:
        signals = ai_assessment.get("signals") if isinstance(ai_assessment, dict) else None
        if isinstance(signals, dict):
            derived_status = _triage_from_ai_signals(signals, facial_symmetry)
            if derived_status is not None:
                triage_status = derived_status

            derived: list[str] = []
            if signals.get("chest_pain") == "present" or signals.get("trouble_breathing") == "present":
                derived.append("Speech indicates a red-flag symptom")
            if signals.get("dizziness") == "present":
                derived.append("Speech indicates dizziness")
            if signals.get("medication_missed") == "present":
                derived.append("Speech indicates medications were missed")

            if derived:
                existing = {str(r).strip().lower() for r in triage_reasons if str(r).strip()}
                for r in derived:
                    if r.strip().lower() not in existing:
                        triage_reasons = list(triage_reasons) + [r]
    except Exception:
        pass

    triage_status_db = triage_status.value.lower()
    checkin["triage_status"] = triage_status

    update_result = get_checkins_collection().update_one(
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
                "ai_assessment": ai_assessment,
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
    if getattr(update_result, "matched_count", 0) != 1:
        raise HTTPException(
            status_code=500,
            detail=f"Check-in completion did not match a DB document for checkin_id={checkin_id}",
        )
    try:
        logger.info(
            "checkin_completed persisted checkin_id=%s modified=%s",
            checkin_id,
            getattr(update_result, "modified_count", None),
        )
    except Exception:
        pass

    return CheckinResult(
        checkin_id=checkin_id,
        triage_status=triage_status,
        triage_reasons=triage_reasons,
        completed_at=checkin["completed_at"],  # type: ignore[arg-type]
    )


@router.get("/{checkin_id}/validate")
def validate_checkin(checkin_id: str):
    """
    Validate a check-in's completeness without completing it.
    
    Returns validation status with any errors or warnings.
    """
    checkin = _load_checkin(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")

    facial_symmetry = None
    if checkin.get("facial_symmetry"):
        facial_symmetry = FacialSymmetryResult(**checkin["facial_symmetry"])

    # Get screening data
    screening = None
    try:
        screening = get_screenings_collection().find_one({"checkin_id": checkin_id})
    except Exception:
        screening = None

    screening_responses = screening.get("responses") if screening else None
    transcript = screening.get("transcript") if screening else None

    # Create empty answers for validation (actual answers come at completion)
    empty_answers = Answers()

    validation_errors = _validate_checkin_completeness(
        checkin=checkin,
        answers=empty_answers,
        facial_symmetry=facial_symmetry,
        transcript=transcript,
        screening_responses=screening_responses,
    )

    blocking_errors = [
        {"field": e.field, "message": e.message}
        for e in validation_errors
        if e.is_blocking
    ]
    warnings = [
        {"field": e.field, "message": e.message}
        for e in validation_errors
        if not e.is_blocking
    ]

    return {
        "checkin_id": checkin_id,
        "is_complete": len(blocking_errors) == 0,
        "errors": blocking_errors,
        "warnings": warnings,
        "has_facial_symmetry": facial_symmetry is not None,
        "has_screening": bool(screening_responses),
        "has_transcript": bool(transcript),
    }


@router.post("/{checkin_id}/auto-complete")
def auto_complete_checkin(checkin_id: str):
    """
    Auto-complete an abandoned check-in that has screening data.
    
    This is useful for check-ins that were started and have screening/facial data
    but were never explicitly completed via the /complete endpoint.
    """
    success = _auto_complete_abandoned_checkin(checkin_id)
    if not success:
        raise HTTPException(
            status_code=422,
            detail="Cannot auto-complete: check-in is already completed or lacks screening data",
        )
    
    return {"checkin_id": checkin_id, "status": "auto-completed"}


@router.post("/cleanup-abandoned")
def cleanup_abandoned_checkins(max_age_hours: int = 24):
    """
    Cleanup abandoned check-ins by auto-completing those with screening data.
    
    Args:
        max_age_hours: Only process check-ins older than this many hours
    
    Returns count of auto-completed check-ins.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    
    # Find in_progress check-ins with screening data
    query = {
        "status": "in_progress",
        "started_at": {"$lte": cutoff},
        "$or": [
            {"transcript": {"$ne": None, "$ne": ""}},
            {"screening_responses": {"$exists": True, "$ne": []}},
        ],
    }
    
    docs = get_checkins_collection().find(query)
    completed_count = 0
    
    for doc in docs:
        checkin_id = doc.get("checkin_id")
        if checkin_id and _auto_complete_abandoned_checkin(checkin_id):
            completed_count += 1
    
    return {
        "auto_completed": completed_count,
        "max_age_hours": max_age_hours,
    }


@router.get("/{checkin_id}", response_model=CheckinDetail)
def get_checkin(checkin_id: str) -> CheckinDetail:
    """Get details of a specific check-in."""
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
        heart_rate=checkin.get("heart_rate"),
        camera_snapshot=checkin.get("camera_snapshot"),
        ai_assessment=checkin.get("ai_assessment"),
    )
