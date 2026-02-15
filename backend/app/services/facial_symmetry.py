"""Facial symmetry analysis service."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from app.models.checkin import FacialSymmetryResult, FacialSymmetrySummary


def _normalize_rollup_values(
    rollups: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Normalize rollup values to ensure consistent float types."""
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
    """Convert facial analysis level to schema format."""
    if not level:
        return "normal"
    lowered = level.lower()
    if lowered == "warning":
        return "warn"
    if lowered == "alert":
        return "alert"
    return lowered


def run_facial_symmetry_analysis(
    video_bytes: bytes, duration_ms: int
) -> FacialSymmetryResult:
    """Run facial symmetry analysis on video bytes."""
    if not video_bytes:
        return FacialSymmetryResult(
            status="ERROR",
            reason="Uploaded camera clip is empty.",
            error="empty_video",
        )

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    try:
        import cv2  # type: ignore
        from run_facial_symmetry_checkin import classify_scientific_index
        from vision.facial_symmetry import FacialSymmetryAnalyzer, summarize_session
    except Exception as exc:
        return FacialSymmetryResult(
            status="SKIPPED",
            reason="Facial symmetry dependencies are not installed; skipping analysis.",
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


def build_facial_symmetry_metrics(
    result: Optional[FacialSymmetryResult],
) -> Optional[Dict[str, Any]]:
    """Build facial symmetry metrics payload for storage."""
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

    return {
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
