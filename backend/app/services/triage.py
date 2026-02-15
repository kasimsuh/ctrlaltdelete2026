"""Triage logic for determining patient status."""

from typing import List, Optional, Tuple

from app.models.checkin import Answers, FacialSymmetryResult, TriageStatus


def compute_triage(answers: Answers) -> Tuple[TriageStatus, List[str]]:
    """Determine triage status based on screening answers."""
    reasons: List[str] = []

    if answers.chest_pain or answers.trouble_breathing:
        reasons.append("Self-reported red flag symptom")
        return TriageStatus.RED, reasons

    if answers.dizziness:
        reasons.append("Reported dizziness")
        return TriageStatus.YELLOW, reasons

    reasons.append("No concerning signals detected")
    return TriageStatus.GREEN, reasons


def merge_triage(
    answers: Answers,
    facial_symmetry: Optional[FacialSymmetryResult],
) -> Tuple[TriageStatus, List[str]]:
    """Merge triage from answers and facial symmetry results."""
    status, reasons = compute_triage(answers)

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
