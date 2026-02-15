from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AnalysisError(Exception):
    """Raised when an uploaded file cannot be analyzed."""


@dataclass
class PipelineOutput:
    hr_bpm: float | None
    sqi: float | None
    engine: str
    note: str | None = None


_MODEL_LOCK = threading.Lock()
_MODEL_INSTANCE: Any | None = None
_TARGET_FPS = 30
_MAX_WIDTH = max(int(os.getenv("OPEN_RPPG_MAX_WIDTH", "640") or "640"), 320)


def analyze_video_file(upload_path: Path, work_dir: Path) -> dict[str, object]:
    if not upload_path.exists():
        raise AnalysisError("Uploaded file was not written to disk.")

    preprocess_started = time.perf_counter()
    prepared_path, prep_note = _prepare_video_for_open_rppg(
        upload_path=upload_path,
        work_dir=work_dir,
    )
    preprocess_ms = round((time.perf_counter() - preprocess_started) * 1000, 1)
    clip_seconds = _estimate_video_duration_seconds(video_path=prepared_path)

    analyze_started = time.perf_counter()
    output = _run_open_rppg(video_path=prepared_path)
    analysis_ms = round((time.perf_counter() - analyze_started) * 1000, 1)

    bpm_series: list[float] = []
    if output.hr_bpm is not None and math.isfinite(output.hr_bpm):
        bpm_series = [round(output.hr_bpm, 2)]

    usable_seconds = round(clip_seconds, 1)
    avg_hr_bpm = round(output.hr_bpm, 1) if output.hr_bpm is not None else None
    hr_quality = _quality_from_metrics(
        avg_hr_bpm=avg_hr_bpm,
        sqi=output.sqi,
        clip_seconds=clip_seconds,
    )

    result: dict[str, object] = {
        "avg_hr_bpm": avg_hr_bpm,
        "hr_quality": hr_quality,
        "usable_seconds": usable_seconds,
        "bpm_series": bpm_series,
        "engine": output.engine,
        "timing_ms": {
            "preprocess": preprocess_ms,
            "analysis": analysis_ms,
        },
    }

    if output.sqi is not None and math.isfinite(output.sqi):
        result["sqi"] = round(output.sqi, 3)

    note = _append_note(output.note, prep_note)
    if clip_seconds <= 0.0:
        note = _append_note(
            note,
            "Could not read clip duration metadata. Check ffmpeg/codec compatibility.",
        )
    if note:
        result["note"] = note

    return result


def _run_open_rppg(video_path: Path) -> PipelineOutput:
    model, note = _get_model()
    if model is None:
        return PipelineOutput(
            hr_bpm=None,
            sqi=None,
            engine="open-rppg",
            note=note or "open-rppg model is unavailable.",
        )

    try:
        results = model.process_video(str(video_path))
    except Exception as exc:
        return PipelineOutput(
            hr_bpm=None,
            sqi=None,
            engine="open-rppg",
            note=f"open-rppg processing failed ({exc.__class__.__name__}: {exc}).",
        )

    if not isinstance(results, dict):
        return PipelineOutput(
            hr_bpm=None,
            sqi=None,
            engine="open-rppg",
            note="open-rppg returned an unexpected response payload.",
        )

    hr = _coerce_float(results.get("hr"))
    sqi = _coerce_float(results.get("SQI"))

    if hr is not None and not (25.0 <= hr <= 240.0):
        hr = None

    note_text = None
    if hr is None:
        note_text = "open-rppg did not return a usable HR value."

    return PipelineOutput(
        hr_bpm=hr,
        sqi=sqi,
        engine="open-rppg",
        note=note_text,
    )


def warmup_open_rppg_model() -> str | None:
    _, note = _get_model()
    return note


def _get_model() -> tuple[Any | None, str | None]:
    global _MODEL_INSTANCE

    with _MODEL_LOCK:
        if _MODEL_INSTANCE is not None:
            return _MODEL_INSTANCE, None

        try:
            import rppg  # type: ignore
        except Exception as exc:
            return None, f"open-rppg import failed ({exc.__class__.__name__}: {exc})."

        model_name = os.getenv("OPEN_RPPG_MODEL", "FacePhys.rlap").strip() or "FacePhys.rlap"
        supported_models = getattr(rppg, "supported_models", None)
        if isinstance(supported_models, list) and model_name not in supported_models:
            return None, (
                f"open-rppg model '{model_name}' is not supported. "
                f"Supported models: {supported_models}."
            )

        init_attempts = (
            {"model": model_name},
            {"model_name": model_name},
            {},
        )
        last_error: Exception | None = None
        for kwargs in init_attempts:
            try:
                _MODEL_INSTANCE = rppg.Model(**kwargs)
                break
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                return None, (
                    "open-rppg model init failed "
                    f"for OPEN_RPPG_MODEL='{model_name}' ({exc.__class__.__name__}: {exc})."
                )

        if _MODEL_INSTANCE is None:
            err = str(last_error) if last_error is not None else "unknown constructor error"
            return None, (
                "open-rppg model init failed "
                f"for OPEN_RPPG_MODEL='{model_name}' (TypeError: {err})."
            )

        return _MODEL_INSTANCE, None


def _quality_from_metrics(
    avg_hr_bpm: float | None,
    sqi: float | None,
    clip_seconds: float,
) -> str:
    if avg_hr_bpm is None:
        return "low"

    if sqi is not None and math.isfinite(sqi):
        if sqi >= 0.75:
            return "high"
        if sqi >= 0.45:
            return "medium"
        return "low"

    if clip_seconds < 10.0:
        return "low"
    if 45.0 <= avg_hr_bpm <= 130.0 and clip_seconds >= 20.0:
        return "high"
    return "medium"


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _append_note(current: str | None, extra: str) -> str:
    if current is None or current.strip() == "":
        return extra
    return f"{current} {extra}"


def _estimate_video_duration_seconds(video_path: Path) -> float:
    ffprobe_duration = _estimate_duration_with_ffprobe(video_path=video_path)
    if ffprobe_duration > 0.0:
        return ffprobe_duration

    try:
        import cv2  # type: ignore
    except Exception:
        return 0.0

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)

    capture.release()

    if fps <= 0.0 or frames <= 0.0:
        return 0.0

    duration = frames / fps
    if not math.isfinite(duration) or duration <= 0.0:
        return 0.0
    return duration


def _estimate_duration_with_ffprobe(video_path: Path) -> float:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return 0.0

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode != 0:
        return 0.0

    raw = completed.stdout.strip()
    if not raw:
        return 0.0

    duration = _coerce_float(raw.splitlines()[0])
    if duration is None or duration <= 0.0:
        return 0.0

    return duration


def _prepare_video_for_open_rppg(upload_path: Path, work_dir: Path) -> tuple[Path, str | None]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        fallback = _convert_to_mp4_if_possible(upload_path=upload_path, work_dir=work_dir)
        return (
            fallback,
            "ffmpeg not found; skipped 30fps normalization (accuracy may decrease).",
        )

    video_filter = (
        f"fps={_TARGET_FPS},"
        f"scale={_MAX_WIDTH}:-2:force_original_aspect_ratio=decrease"
    )
    prepared_path = work_dir / "open_rppg_input.mp4"
    command_primary = [
        ffmpeg_path,
        "-y",
        "-i",
        str(upload_path),
        "-an",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(_TARGET_FPS),
        "-keyint_min",
        str(_TARGET_FPS),
        "-sc_threshold",
        "0",
        str(prepared_path),
    ]
    if _run_ffmpeg(command_primary) and prepared_path.exists() and prepared_path.stat().st_size > 0:
        return prepared_path, None

    # Fallback for ffmpeg builds without libx264.
    command_fallback = [
        ffmpeg_path,
        "-y",
        "-i",
        str(upload_path),
        "-an",
        "-vf",
        video_filter,
        "-c:v",
        "mpeg4",
        "-q:v",
        "4",
        "-g",
        str(_TARGET_FPS),
        str(prepared_path),
    ]
    if _run_ffmpeg(command_fallback) and prepared_path.exists() and prepared_path.stat().st_size > 0:
        return prepared_path, None

    fallback = _convert_to_mp4_if_possible(upload_path=upload_path, work_dir=work_dir)
    return (
        fallback,
        "Failed to normalize clip to 30fps MP4; using original container (accuracy may decrease).",
    )


def _run_ffmpeg(command: list[str]) -> bool:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _convert_to_mp4_if_possible(upload_path: Path, work_dir: Path) -> Path:
    if upload_path.suffix.lower() == ".mp4":
        return upload_path

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return upload_path

    converted_path = work_dir / "upload.mp4"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(upload_path),
        str(converted_path),
    ]

    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return upload_path

    if not converted_path.exists() or converted_path.stat().st_size == 0:
        return upload_path

    return converted_path
