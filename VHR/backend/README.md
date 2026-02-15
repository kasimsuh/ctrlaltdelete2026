# VHR Backend (open-rppg)

FastAPI backend for the VHR spike.

`POST /rppg` accepts an uploaded webcam clip and runs the
[`open-rppg`](https://github.com/KegangWangCCNU/open-rppg) video pipeline.

The frontend contract is unchanged:

- `avg_hr_bpm`
- `hr_quality`
- `usable_seconds`
- `bpm_series`
- `engine`
- optional `note`

## What changed

- Removed `custom_rppg` and `pyVHR` engine paths.
- Replaced analyzer engine with `open-rppg` (`rppg.Model().process_video(...)`).
- Kept route and response shape so `VHR/frontend` still works.

## Dependencies

From `requirements.txt`:

- `fastapi`
- `uvicorn[standard]`
- `python-multipart`
- `numpy`
- `opencv-python`
- `open-rppg`

System tool:

- `ffmpeg` + `ffprobe` (required for stable preprocessing of browser uploads)

## Setup

Install system dependency first (macOS):

```bash
brew install ffmpeg
```

Ensure your shell can see Homebrew binaries (zsh):

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
eval "$(/opt/homebrew/bin/brew shellenv)"
hash -r
ffmpeg -version
ffprobe -version
```

Create backend env and install Python deps:

```bash
cd VHR/backend
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional model override:

```bash
export OPEN_RPPG_MODEL=FacePhys.rlap
```

## Run backend

```bash
cd VHR/backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## API

### `GET /health`

```json
{ "ok": true }
```

### `POST /rppg`

- `multipart/form-data`
- field name: `video`

Example:

```bash
curl -X POST -F "video=@/path/to/session.webm" http://localhost:8000/rppg
```

Example response:

```json
{
  "avg_hr_bpm": 72.6,
  "hr_quality": "high",
  "usable_seconds": 24.8,
  "bpm_series": [72.6],
  "engine": "open-rppg",
  "sqi": 0.84
}
```

## Notes

- First model init can be slower than subsequent requests.
- `open-rppg` can return no HR on very short clips, motion-heavy clips, or poor lighting.
- In those cases, response keeps the same shape with `avg_hr_bpm: null` and a `note`.
- `open-rppg` is sensitive to non-keyframe / variable-fps recordings (common with browser `MediaRecorder`).
  The backend now tries to re-encode uploads to keyframe-only 30fps via `ffmpeg`; install `ffmpeg`
  to enable that path.

## Troubleshooting

- If `ffmpeg` or `ffprobe` says `command not found`, your PATH is not initialized for Homebrew in that shell.
  Run:
  `eval "$(/opt/homebrew/bin/brew shellenv)"` and restart terminal/VS Code terminal.
- macOS warning about duplicate `AVFFrameReceiver` / `AVFAudioReceiver` classes from `cv2` and `av`:
  `open-rppg` imports both PyAV (`av`) and OpenCV (`cv2`) in the same process, which can print this warning on macOS.
  It is usually non-fatal, but if you see crashes, run backend on Linux (Docker/VM) where this Objective-C conflict does not apply.
