from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import HTTPException, UploadFile

from .analyzer import AnalysisError, analyze_video_file

_UPLOAD_CHUNK_SIZE = 1024 * 1024


async def analyze_uploaded_video(video: UploadFile) -> dict[str, object]:
    filename = video.filename or "checkin.webm"
    suffix = Path(filename).suffix.lower() or ".webm"

    request_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="checkin-vhr-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        upload_path = tmp_root / f"upload{suffix}"

        write_started = time.perf_counter()
        bytes_written = await _write_upload_file(video=video, upload_path=upload_path)
        upload_write_ms = round((time.perf_counter() - write_started) * 1000, 1)

        if bytes_written <= 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty.")

        try:
            result = analyze_video_file(upload_path=upload_path, work_dir=tmp_root)
        except AnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"VHR analysis failed: {exc}") from exc

    total_ms = round((time.perf_counter() - request_started) * 1000, 1)
    timing = result.get("timing_ms")
    if isinstance(timing, dict):
        timing["upload_write"] = upload_write_ms
        timing["total"] = total_ms
    else:
        result["timing_ms"] = {
            "upload_write": upload_write_ms,
            "total": total_ms,
        }
    result["upload_mb"] = round(bytes_written / 1024 / 1024, 2)

    return result


async def _write_upload_file(video: UploadFile, upload_path: Path) -> int:
    bytes_written = 0
    with upload_path.open("wb") as output:
        while True:
            chunk = await video.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            bytes_written += len(chunk)
    await video.close()
    return bytes_written
