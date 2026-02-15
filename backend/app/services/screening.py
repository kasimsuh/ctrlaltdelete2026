"""Screening processing service."""

from typing import List

from app.models.screening import ScreeningResponseItem


def build_screening_transcript(responses: List[ScreeningResponseItem]) -> str:
    """Build a transcript from screening responses."""
    lines: List[str] = []
    for item in responses:
        if item.q:
            lines.append(f"AI: {item.q}")
        if item.transcript:
            lines.append(f"USER: {item.transcript}")
    return " ".join(lines)
