# VHR Frontend

Frontend app for the VHR spike demo.

This app runs a single-page webcam flow:

1. Start camera (`getUserMedia`)
2. Record clip (`MediaRecorder`)
3. Upload clip to backend (`POST /rppg`)
4. Log returned rPPG values to browser console

It is intentionally minimal and optimized for proving end-to-end compatibility with the backend spike.

## Directory Contents

- `app/page.tsx`: Entire demo UI and behavior (camera, recording, upload, result display).
- `app/layout.tsx`: Root layout metadata and global stylesheet import.
- `app/globals.css`: Styling, typography, responsive layout, and lightweight animation.
- `package.json`: Next.js scripts and dependencies.
- `tsconfig.json`: TypeScript compiler settings.
- `next.config.mjs`: Next.js config (`reactStrictMode` enabled).
- `next-env.d.ts`: Next.js type reference file.
- `.env`: Local environment vars (`NEXT_PUBLIC_BACKEND_URL`).
- `.next/`: Build/dev artifacts (generated).
- `node_modules/`: Installed packages (generated).

## Tech Stack

- Next.js `14.2.5` (App Router)
- React `18.3.1`
- TypeScript `5.5.x`
- Browser APIs:
  - `navigator.mediaDevices.getUserMedia`
  - `MediaRecorder`
  - `FormData`
  - `fetch`

## App Behavior

### High-Level Flow

1. User clicks `Start Camera`.
2. Browser requests camera+microphone permissions.
3. Live stream is attached to `<video>` preview.
4. User clicks `Start Recording`.
5. Chunks are buffered while recording.
6. User clicks `Stop Recording`.
7. Chunks are combined into a `Blob`.
8. If auto-upload is on, clip is immediately sent to backend.
9. Backend response is:
   - logged to browser console (`console.log("rPPG result", json)`)
   - summarized in the on-page JSON preview block.

### UI Controls and Guardrails

Buttons are intentionally constrained to avoid invalid states:

- `Start Camera` is disabled after stream is active or during upload.
- `Start Recording` requires a stream and no active recording/upload.
- `Stop Recording` requires active recording and no active upload.
- `Upload Clip` requires a finished clip and no active recording/upload.

Status text (`statusMessage`) gives immediate user feedback for each transition.

## Detailed Implementation Notes (`app/page.tsx`)

### Core State

- `stream: MediaStream | null`
  - Active webcam+mic stream used for preview and recording.
- `isRecording: boolean`
  - Whether recorder is currently capturing.
- `recordedBlob: Blob | null`
  - Final clip after stop.
- `isUploading: boolean`
  - Request in-flight state for upload.
- `autoUpload: boolean`
  - If true, upload runs from recorder `onstop`.
- `statusMessage: string`
  - User-facing state text.
- `lastResult: RppgResponse | null`
  - Last backend response shown in UI.

### Refs

- `videoRef`
  - DOM ref to preview element.
- `mediaRecorderRef`
  - Mutable recorder instance for start/stop control.
- `chunksRef`
  - Buffer for streamed recording chunks.

### MIME Type Selection

`getSupportedMimeType()` tests support in this order:

1. `video/webm;codecs=vp9,opus`
2. `video/webm;codecs=vp8,opus`
3. `video/webm`
4. `video/mp4`

First supported type is used. If none are explicitly supported, recorder falls back to default constructor behavior.

### Camera Setup

`startCamera()` calls:

- `getUserMedia({ video: { facingMode: "user", width: ideal 1280, height: ideal 720 }, audio: true })`

When stream is available, a React effect binds `videoRef.current.srcObject = stream` and calls `play()`.

Cleanup effect stops all tracks on unmount or stream replacement.

### Recording

`startRecording()`:

- Initializes `MediaRecorder` from current stream.
- Clears `chunksRef`.
- On every `dataavailable`, appends non-empty chunk.
- On `stop`, combines chunks into one Blob with recorder mime type.
- Stores `recordedBlob` and updates status.
- Optionally auto-uploads.

Recorder is started with `recorder.start(1000)`, requesting chunk emission roughly every second.

### Upload

`uploadClip()`:

- Uses passed blob or last recorded blob.
- Chooses filename extension from blob type (`.mp4` or `.webm`).
- Appends to `FormData` under `video`.
- Sends `POST ${BACKEND_URL}/rppg`.
- On non-OK response, throws with returned body text.
- On success:
  - logs full payload to console
  - stores result in `lastResult`
  - updates status message.

### Backend Response Type

Local frontend type:

```ts
type RppgResponse = {
  avg_hr_bpm: number | null;
  hr_quality: "low" | "medium" | "high";
  usable_seconds: number;
  bpm_series?: number[];
  engine?: string;
  note?: string;
};
```

The on-page `<pre>` currently shows a compact subset:

- `avg_hr_bpm`
- `hr_quality`
- `usable_seconds`
- `engine`

Full payload is always available in browser console.

## Styling (`app/globals.css`)

Design choices:

- Gradient + radial layered background to avoid flat default look.
- Glass-style card container (`backdrop-filter: blur`).
- Two-column desktop layout with mobile collapse at `900px`.
- Accent pulse animation for Gemini placeholder indicator.
- Custom font pairing:
  - `Space Grotesk` for UI text
  - `DM Mono` for metadata and JSON block.

No UI framework is used. All styles are global CSS for speed and simplicity.

## Environment and Configuration

### `.env`

Current variable:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

This is read in `page.tsx`:

- default fallback: `http://localhost:8000`
- override by setting `NEXT_PUBLIC_BACKEND_URL`

### Next config

`next.config.mjs`:

- `reactStrictMode: true`

## Local Development

### Prerequisites

- Node.js 20 LTS (`.nvmrc` is set to `20`)
- npm
- Running backend at configured URL

### Install

```bash
cd VHR/frontend
npm install
```

### Run dev server

```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`.

### Build and run production mode

```bash
npm run build
npm run start
```

## Browser and Device Constraints

- Camera/microphone access requires a secure context:
  - `http://localhost` is allowed
  - non-localhost plain `http` generally is not
- User must grant camera and mic permissions.
- `MediaRecorder` codec support varies by browser.
- Mobile browsers may pause background capture if tab/app loses focus.

Recommended for demo reliability:

- Use latest Chrome desktop.
- Record 20-30 seconds.
- Keep face centered and lighting stable.

## Manual Test Checklist

1. Start backend and confirm `GET /health` works.
2. Open frontend.
3. Click `Start Camera` and approve permissions.
4. Confirm live preview appears.
5. Click `Start Recording`, wait ~20-30s, click `Stop Recording`.
6. Confirm upload happens automatically (if toggle on).
7. Confirm browser console logs `rPPG result`.
8. Confirm on-page JSON block updates.
9. Toggle auto-upload off and verify manual `Upload Clip` works.

## Troubleshooting

### Camera does not start

- Verify browser permission has not been blocked.
- Ensure another app is not exclusively using the camera.
- Check console for `getUserMedia` errors.

### Recording starts but upload fails

- Verify backend is running at `NEXT_PUBLIC_BACKEND_URL`.
- Confirm CORS/network path is reachable from browser.
- Check backend logs for request and error details.

### Upload returns non-OK status

- Frontend surfaces backend response text in thrown error.
- Inspect browser DevTools Network tab for response payload.
- Inspect backend stderr/stdout for analyzer errors.

### `TypeError: Telemetry is not a constructor` or `@edge-runtime/primitives` startup errors

- This project's pinned `next@14.2.5` is unstable on Node 22 in this setup.
- Switch to Node 20 and reinstall dependencies:
  - `cd VHR/frontend`
  - `nvm install 20 && nvm use 20`
  - `rm -rf node_modules .next package-lock.json`
  - `npm install`
  - `npm run dev`

### No BPM or low quality

- This can be expected with poor clip conditions.
- Ensure clip is long enough and face is stable.
- Confirm backend response `engine`:
  - `open-rppg` for the backend estimator

## Extension Points

- Replace Gemini placeholder with real live conversation UI/audio.
- Add recording timer and elapsed indicator.
- Add retry/backoff and richer upload error UI.
- Add explicit state machine to reduce implicit transitions.
- Split `page.tsx` into:
  - `components/RecorderControls.tsx`
  - `hooks/useRecorder.ts`
  - `services/upload.ts`
- Add unit tests for state transitions and upload handling.

## Current Scope Boundaries

- No auth/session handling.
- No persistent frontend storage.
- No waveform/audio visualization.
- No multi-page routes.
- No formal test suite yet.

This is a focused spike frontend for proving webcam-record-upload-analyze flow.
