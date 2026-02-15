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
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m uvicorn app.main:app --reload --port 8000
```

3. MongoDB (optional, with auth)

```bash
cp .env.example .env
docker compose up -d mongo
```
