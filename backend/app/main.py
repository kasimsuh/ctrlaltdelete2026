"""
Guardian Check-In API

FastAPI application entry point with minimal setup.
All routes, models, and business logic are organized in separate modules.
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import ensure_user_indexes
from app.routes import register_routes

# Load environment from backend/.env regardless of launch directory
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="Guardian Check-In API", version="0.1.0")

# CORS configuration for frontend development servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routes
register_routes(app)


@app.on_event("startup")
def _startup() -> None:
    """Initialize database indexes on startup."""
    try:
        ensure_user_indexes()
    except Exception:
        # Best-effort index creation; auth still works without it
        pass
