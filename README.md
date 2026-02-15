# SeniCare

Senior-friendly daily check-in app with:

- webcam capture and facial symmetry analysis
- optional video heart-rate estimation (VHR / open-rppg)
- live voice screening (Gemini Live) + transcript parsing
- triage classification (`Green`, `Yellow`, `Red`)
- doctor dashboard and AI-generated senior summary

This repository contains:

- main product app (`frontend/` + `backend/`)
- a separate VHR spike prototype (`VHR/`) kept for reference

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Tech Stack](#tech-stack)
4. [How the App Works](#how-the-app-works)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Database and Data Model](#database-and-data-model)
9. [Frontend Notes](#frontend-notes)
10. [Backend Notes](#backend-notes)
11. [Testing and Validation](#testing-and-validation)
12. [Troubleshooting](#troubleshooting)
13. [Additional Docs](#additional-docs)
14. [Current Limitations](#current-limitations)

## Project Overview

The main app provides two role-based experiences:

- `senior`: daily check-in flow with camera + voice screening
- `doctor`: dashboard analytics, senior list, and AI summary generation

Primary flow (senior):

1. user starts check-in (`POST /checkins/start`)
2. frontend records ~10s camera clip and uploads it (`POST /checkins/{id}/upload`)
3. backend runs facial symmetry + VHR analysis
4. live voice screening captures responses
5. frontend posts responses/transcript (`POST /screenings`, then `POST /checkins/{id}/complete`)
6. backend computes triage and persists result

## Repository Layout

```text
.
├── frontend/                     # Vite + React client (main app)
│   ├── src/
│   │   ├── components/           # Login, SeniorCheckin, DoctorDashboard
│   │   ├── hooks/                # useAuth, useCheckin, useDoctorDashboard
│   │   └── lib/                  # API helpers, audio helpers, screening constants
│   ├── public/senicarelogo.png
│   └── .env.example
├── backend/                      # FastAPI API (main app)
│   └── app/
│       ├── routes/               # auth, checkins, dashboard, seniors, reports...
│       ├── services/             # triage, AI summary, facial symmetry
│       ├── models/               # Pydantic request/response schemas
│       ├── vhr/                  # open-rppg video HR pipeline
│       ├── auth.py               # JWT + password auth logic
│       ├── db.py                 # Mongo connection helpers
│       └── main.py               # app setup + route registration
├── vision/                       # standalone facial symmetry analyzer module
├── run_facial_symmetry_checkin.py# local CLI diagnostic capture script
├── mongo/init-mongo.js           # Mongo init helper script
├── ARCHITECTURE.md               # high-level architecture notes
├── MONGODB_SCHEMA_SIMPLE.md      # schema and query examples
└── VHR/                          # separate spike demo (Next.js + FastAPI)
```

## Tech Stack

### Frontend (main app)

- React 18 + Vite 5
- Tailwind CSS
- `@google/genai` for Gemini Live session from browser
- browser APIs: `MediaRecorder`, `getUserMedia`, `SpeechRecognition`

### Backend (main app)

- FastAPI + Uvicorn
- MongoDB + PyMongo
- JWT auth (`python-jose`) + password hashing (`passlib`)
- Gemini server SDK (`google-genai`) for ephemeral token + report summaries
- MediaPipe + OpenCV for facial symmetry
- `open-rppg` + `ffmpeg/ffprobe` for video heart-rate estimation

## How the App Works

### Authentication flow

- Register/Login endpoints issue JWT access tokens.
- Frontend stores JWT in `localStorage` (`guardian_checkin.jwt`).
- `/me` returns role/profile used to route UI:
  - `doctor` -> Doctor Dashboard
  - others -> Senior Check-in

### Senior check-in flow

- `startVoice()` in `useCheckin` orchestrates:
  - create check-in
  - request ephemeral Gemini token from backend
  - connect Gemini Live audio session
  - record/upload 10s camera clip
  - collect speech responses
  - submit screening + complete check-in
- Triage logic combines symptom answers and facial symmetry status.

### Doctor flow

- Dashboard pulls:
  - `/dashboard/analytics`
  - `/dashboard/seniors`
- Doctor can request AI summary:
  - `POST /reports/senior-summary`

## Quick Start

## 1) Prerequisites

- Node.js 18+ (20 recommended)
- Python 3.10+
- MongoDB instance (local or Atlas)
- `ffmpeg` + `ffprobe` installed (recommended for VHR reliability)

macOS:

```bash
brew install ffmpeg
```

## 2) Backend setup (Terminal A)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`.

## 3) Frontend setup (Terminal B)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

## 4) Verify

- `GET http://localhost:8000/health`
- open `http://localhost:5173`

## 5) Create a doctor user (optional but needed for dashboard)

Registration creates `senior` users by default. To access doctor routes/UI, update a user role in Mongo:

```javascript
use guardian
db.users.updateOne(
  { email: "doctor@example.com" },
  { $set: { role: "doctor" } }
)
```

## Configuration

## Frontend env

File: `frontend/.env.example`

```env
VITE_API_BASE=http://localhost:8000
```

## Backend env

No checked-in `backend/.env.example` currently. Create `backend/.env` (or set shell env vars) with:

```env
# Mongo
MONGODB_URI=mongodb://localhost:27017/guardian
# or use split fields below instead of MONGODB_URI:
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=guardian
MONGO_USER=
MONGO_PASSWORD=
MONGO_AUTH_SOURCE=guardian
MONGO_SERVER_SELECTION_TIMEOUT_MS=5000
MONGO_CONNECT_TIMEOUT_MS=5000

# Auth
JWT_SECRET=replace-with-a-strong-secret
JWT_ALG=HS256
JWT_EXPIRES_MIN=60
REQUIRE_AUTH=false

# Gemini
GEMINI_API_KEY=replace-with-real-key

# VHR/open-rppg (optional tuning)
OPEN_RPPG_MODEL=FacePhys.rlap
OPEN_RPPG_MAX_WIDTH=640
```

Notes:

- `JWT_SECRET` is required for token issuance/verification.
- `GEMINI_API_KEY` is required for:
  - `/auth/ephemeral`
  - `/reports/senior-summary` AI summaries

## API Reference

Base URL: `http://localhost:8000`

## Health

- `GET /health`
  - returns app health, mongo status, auth flags, runtime metadata

## Auth

- `POST /auth/register`
  - body: `{ firstName, lastName, email, password }`
- `POST /auth/login`
  - body: `{ email, password }`
  - returns `{ access_token, token_type }`
- `GET /me`
  - requires bearer token
- `GET /auth/ephemeral`
- `POST /auth/ephemeral`
  - returns short-lived Gemini token for frontend live session

## Check-ins

- `POST /checkins/start` (auth required)
  - creates in-progress check-in
- `POST /checkins/{checkin_id}/upload`
  - multipart upload
  - accepts `video`, optional `audio`, optional `frames[]`, optional `metadata`
  - runs facial symmetry + VHR analysis
- `POST /checkins/{checkin_id}/complete`
  - body: `{ answers, transcript }`
  - validates completeness and computes triage
- `GET /checkins/{checkin_id}`
  - returns detailed check-in model
- `GET /checkins/{checkin_id}/validate`
  - pre-completion data completeness check
- `POST /checkins/{checkin_id}/auto-complete`
  - auto-complete abandoned in-progress session with existing data
- `POST /checkins/cleanup-abandoned?max_age_hours=24`
  - bulk auto-complete stale sessions

## Screenings

- `POST /screenings`
  - create screening session linked to check-in
- `GET /screenings/{session_id}`
  - reads in-memory screening cache (not persistent lookup)

## Seniors

- `GET /seniors/{senior_id}/checkins`
  - query params: `from_date`, `to_date`, `include_incomplete`
- `GET /seniors/{senior_id}/baseline`
- `POST /seniors/{senior_id}/summaries/weekly`
- `GET /seniors/{senior_id}/summaries/weekly?week_start=...`
- `GET /seniors/{senior_id}/alerts`

## Dashboard (doctor role required)

- `GET /dashboard/analytics`
- `GET /dashboard/seniors`

## Reports (doctor role required)

- `POST /reports/senior-summary`
  - AI-generated summary from provided overview/checkins payload

## Alerts

- `POST /alerts/test`
  - creates test alert record in memory

## Database and Data Model

Main collections used by backend:

- `users`
- `checkin_history`
- `screenings`

Key points:

- Check-ins are persisted in Mongo (`checkin_history`).
- Some app data is intentionally in-memory in current implementation:
  - alerts store (`ALERTS`)
  - weekly summaries (`WEEKLY_SUMMARIES`)
  - baseline cache (`BASELINES`)
  - screening GET cache (`SCREENINGS`)
- In-memory data resets on backend restart.

See:

- `MONGODB_SCHEMA_SIMPLE.md` for detailed sample documents and indexes.

## Frontend Notes

- API base comes from `VITE_API_BASE`.
- Login/registration UI is in `frontend/src/components/login/*`.
- Registration currently enforces `@gmail.com` client-side in `useAuth`.
- Senior check-in experience is implemented in:
  - `frontend/src/components/SeniorCheckin.jsx`
  - `frontend/src/hooks/useCheckin.js`
- Doctor dashboard/report UI is in:
  - `frontend/src/components/DoctorDashboard.jsx`
  - `frontend/src/hooks/useDoctorDashboard.js`
- Favicon and brand asset:
  - `frontend/public/senicarelogo.png`

## Backend Notes

- Entry point: `backend/app/main.py`
- Routes registered from `backend/app/routes/__init__.py`
- CORS allows:
  - `http://localhost:5173`
  - `http://localhost:5174`
  - `http://localhost:4173`
- User index creation (`users.email` unique) is attempted on startup.

## Testing and Validation

Current repo state:

- No top-level automated test command is configured in `frontend/package.json` or root.
- Manual validation recommended:
  1. start backend/frontend
  2. register/login
  3. run check-in with camera + mic permissions
  4. verify `/checkins/*` writes in Mongo
  5. test doctor dashboard with a `doctor` role user

## Troubleshooting

## `JWT_SECRET is not set`

- Set `JWT_SECRET` in backend env before using `/auth/login` and protected routes.

## Mongo connection errors on `/health`

- Verify `MONGODB_URI` or `MONGO_*` values.
- Ensure network access/credentials are correct.

## `/auth/ephemeral` fails

- Set a valid `GEMINI_API_KEY`.

## VHR returns low quality or null BPM

- Use longer, stable, well-lit clips.
- Install `ffmpeg`/`ffprobe`.
- Reduce motion and keep face centered.

## Camera/microphone issues in browser

- Use HTTPS or `localhost`.
- Grant browser permissions for camera and microphone.
- Prefer latest Chrome for most stable `MediaRecorder` and speech APIs.

## Additional Docs

- `ARCHITECTURE.md`: high-level product and architecture writeup
- `MONGODB_SCHEMA_SIMPLE.md`: practical schema + indexes + query examples
- `backend/README.md`: backend-specific setup details
- `VHR/README.md`: standalone VHR spike walkthrough

## Standalone Facial Symmetry CLI

For local terminal diagnostics outside the web app:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_facial_symmetry_checkin.py --duration 20 --show-video
```

This uses `vision/facial_symmetry.py` and the MediaPipe task model at `models/face_landmarker.task` (auto-download fallback is implemented).

## Contributors

- Daniel Zhong
- Raihan Carder
- Suhiyini Kasim
- Long Vo
