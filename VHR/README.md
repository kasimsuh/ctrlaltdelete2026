# VHR Spike Demo (open-rppg)

Minimal end-to-end demo for the `plan.md` flow:

- Frontend (`frontend/`): Next.js page that records webcam + mic with `MediaRecorder`.
- Backend (`backend/`): FastAPI endpoint that accepts video upload and runs `open-rppg`.

## Run It

Use two terminals: one for backend, one for frontend.

### 1) Start backend (Terminal A)

```bash
brew install ffmpeg

# if brew binaries are not visible in your shell
eval "$(/opt/homebrew/bin/brew shellenv)"
ffmpeg -version
ffprobe -version

cd VHR/backend
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"ok":true}
```

### 2) Start frontend (Terminal B)

```bash
cd VHR/frontend
nvm use || nvm install
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`.

## Video Demo Flow

1. Click `Start Camera`.
2. Click `Start Recording` and record about 20-30 seconds.
3. Click `Stop Recording`.
4. Clip auto-uploads by default (or click `Upload Clip`).
5. The backend returns JSON from `POST /rppg`.

## JSON Returned For a Video Upload

The backend response includes these main fields:

- `avg_hr_bpm`: average estimated heart rate (`number | null`)
- `hr_quality`: quality label (`"low" | "medium" | "high"`)
- `usable_seconds`: analyzed clip duration in seconds
- `bpm_series`: per-window BPM values (currently a single value when available)
- `engine`: estimator name (`"open-rppg"`)
- Optional: `sqi`, `note`, `timing_ms`, `upload_mb`

Example:

```json
{
  "avg_hr_bpm": 72.6,
  "hr_quality": "high",
  "usable_seconds": 24.8,
  "bpm_series": [72.6],
  "engine": "open-rppg",
  "sqi": 0.84,
  "timing_ms": {
    "upload_write": 118.4,
    "preprocess": 936.5,
    "analysis": 1624.8,
    "total": 2717.3
  },
  "upload_mb": 4.21
}
```

If the model cannot extract a usable signal, `avg_hr_bpm` can be `null` and `note` explains why.

## Notes

- `ffmpeg`/`ffprobe` are strongly recommended for stable preprocessing (webm/mp4 normalization).
