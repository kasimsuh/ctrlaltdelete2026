# Guardian Check-In API (FastAPI)

## Quick start

1. Create a virtual environment and install deps.
2. Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at http://localhost:8000.

## Video HR (VHR)

`POST /checkins/{checkin_id}/upload` now analyzes the uploaded `video` file
with the backend VHR module (`app/vhr`) and returns HR output in `vhr`:

- `avg_hr_bpm`
- `hr_quality`
- `usable_seconds`
- `bpm_series`
- `engine`
- optional `sqi`, `note`, `timing_ms`, `upload_mb`

Additional VHR dependencies are listed in `backend/requirements.txt`.
For stable browser video preprocessing, install `ffmpeg`/`ffprobe` on the host.

## MongoDB (local docker-compose)

This repo includes a `docker-compose.yml` that starts MongoDB with authentication enabled and creates an app user on first run.

1. Copy `.env.example` to `.env` and set strong passwords.
2. Start MongoDB:

```bash
docker compose up -d mongo
```

3. Configure backend connection (either set `MONGODB_URI` or the `MONGO_*` fields in `.env`):

- `MONGO_HOST`, `MONGO_PORT`
- `MONGO_DB`
- `MONGO_USER`, `MONGO_PASSWORD`
- `MONGO_AUTH_SOURCE` (defaults to `MONGO_DB`)

The `/health` endpoint reports Mongo connectivity in the `mongo` field.

The backend auto-loads `.env` from either `backend/.env` or the repo root `.env` when starting `app.main`.

## User Auth (JWT)

Auth endpoints:

- `POST /auth/register` -> create user
- `POST /auth/login` -> returns `{ access_token }`
- `GET /me` -> requires `Authorization: Bearer <token>`

Environment variables:

- `JWT_SECRET` (required for login/token verification)
- `JWT_ALG` (default `HS256`)
- `JWT_EXPIRES_MIN` (default `60`)
- `REQUIRE_AUTH` (default `false`; when `true`, missing tokens on protected endpoints should be rejected once you add enforcement to routes)
