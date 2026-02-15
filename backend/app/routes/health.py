"""Health check endpoint."""

import os
from datetime import datetime

from fastapi import APIRouter

from app.db import mongo_check
from app.models.health import HealthStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
def health_check() -> HealthStatus:
    """Check API and database health status."""
    ok, summary, err = mongo_check()
    return HealthStatus(
        time=datetime.utcnow(),
        mongo="ok" if ok else "error",
        mongo_host=summary.get("host"),
        mongo_db=summary.get("db"),
        mongo_error=err,
        auth_required=(
            os.environ.get("REQUIRE_AUTH", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        jwt_configured=bool(os.environ.get("JWT_SECRET")),
    )
