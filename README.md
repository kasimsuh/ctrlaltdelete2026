# Guardian Check-In

Boilerplate for the hackathon MVP: Vite frontend + FastAPI backend.

## Structure

- frontend: Vite web client
- backend: FastAPI API server

## Run locally

1. Frontend

```bash
cd frontend
npm install
npm run dev
```

2. Backend

```bash
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

3. MongoDB (optional, with auth)

```bash
cp .env.example .env
docker compose up -d mongo
```
