# plan.md — Spike: Record Live Video -> Upload Clip -> open-rppg Avg HR -> Console Output

## Goal

Create a working spike (not full product) that proves this flow works:

1. A user has a live webcam session while "talking with Gemini" (Gemini can be a stub; webcam recording is the key).
2. The app records the session (video+audio) into a clip.
3. The user clicks Upload (or upload automatically on stop).
4. A Python backend runs the `open-rppg` analyzer on the clip and returns:
   - `avg_hr_bpm`
   - `hr_quality` (low/medium/high)
   - `usable_seconds`
5. The result is printed to:
   - browser console (Next.js)
   - server console (FastAPI)

Success = upload a clip, receive HR from `open-rppg`, and log output.

## Non-goals

- No auth, MongoDB, Twilio, clinician summary, or baseline logic
- No UI polish (buttons + preview is enough)
- Gemini does not need to be real in this spike

## Assumptions / Constraints

- Face should be centered and well lit.
- Clip should be at least 15–30s with stable framing.
- Motion-heavy clips may return low quality.

## High-Level Architecture

### Frontend (Next.js)

- Uses `getUserMedia()` for preview (`<video>`).
- Uses `MediaRecorder` to capture webcam stream to blob.
- Uploads clip to backend endpoint `POST /rppg`.
- Logs backend response to browser console.

### Backend (FastAPI + open-rppg)

- Accepts `multipart/form-data` upload.
- Converts container if needed (`webm -> mp4`) via ffmpeg.
- Runs `open-rppg` on uploaded video and reads:
  - `hr` (heart rate)
  - `SQI` (signal quality index, when available)
- Computes:
  - `avg_hr_bpm`
  - `hr_quality`
  - `usable_seconds`
- Returns JSON and logs to server console.

## Repo Layout

```
root/
  frontend/
  backend/
  sample_clips/
  plan.md
```

## Frontend Plan (Next.js)

### UX

Single page:

- Start Camera
- Start Recording
- Stop Recording
- Upload (optional when auto-upload is enabled)
- Webcam preview

### Implementation Steps

1. Implement webcam preview with `navigator.mediaDevices.getUserMedia({ video: true, audio: true })`.
2. Implement recording with `MediaRecorder` and chunk buffer.
3. On stop, build blob and upload via `FormData` to `/rppg`.
4. Print response in browser console.

Acceptance test:

- Record 20–30s while speaking.
- Upload returns JSON.
- Console shows `avg_hr_bpm` and `hr_quality`.

## Backend Plan (FastAPI + open-rppg)

### Install & Environment

- Install FastAPI + open-rppg dependencies.
- Install ffmpeg (recommended for conversion reliability).

### Endpoint

POST `/rppg`

- input: multipart field `video`
- output:
  - `avg_hr_bpm`: number | null
  - `hr_quality`: "low" | "medium" | "high"
  - `usable_seconds`: number
  - `bpm_series`: number[]
  - `engine`: "open-rppg"

### Processing Steps

1. Receive file and save to temp path.
2. Convert to mp4 if needed with ffmpeg.
3. Run `open-rppg` pipeline on the video.
4. Compute avg HR from BPM series.
5. Compute quality label using basic heuristics.
6. Log values in server console and return JSON.

Acceptance test:

- Upload clip with centered face + stable light.
- Backend prints computed values.
- Response includes non-null `avg_hr_bpm` when signal quality is sufficient.

## Edge Cases

- Motion / poor lighting: `avg_hr_bpm` may be null or quality low.
- Very short clips (<10s): return null/low.
- Multiple faces: results may be unstable in spike mode.

## Done Checklist

- [ ] Next.js previews webcam and records clip.
- [ ] Clip uploads successfully to FastAPI.
- [ ] FastAPI runs `open-rppg` and returns HR output.
- [ ] Backend returns `{ avg_hr_bpm, hr_quality, usable_seconds }`.
- [ ] Browser console prints return values.
- [ ] Server console prints computed values.
