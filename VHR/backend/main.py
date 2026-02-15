from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analyzer import AnalysisError, analyze_video_file, warmup_open_rppg_model

app = FastAPI(title="VHR open-rppg Backend", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _write_upload_file(video: UploadFile, upload_path: Path) -> int:
    bytes_written = 0
    with upload_path.open("wb") as output:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            bytes_written += len(chunk)
    await video.close()
    return bytes_written


@app.on_event("startup")
def _warmup_model() -> None:
    note = warmup_open_rppg_model()
    if note:
        print(f"[rPPG] model warmup warning: {note}")
        return
    print("[rPPG] model warmup ready.")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/rppg")
async def rppg(video: UploadFile = File(...)) -> dict[str, object]:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    request_started = time.perf_counter()
    suffix = Path(video.filename).suffix.lower() or ".webm"

    with tempfile.TemporaryDirectory(prefix="vhr-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        upload_path = tmp_root / f"upload{suffix}"

        write_started = time.perf_counter()
        bytes_written = await _write_upload_file(video=video, upload_path=upload_path)
        upload_write_ms = round((time.perf_counter() - write_started) * 1000, 1)

        if bytes_written <= 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            result = analyze_video_file(upload_path=upload_path, work_dir=tmp_root)
        except AnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

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

    print("[rPPG]", json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
