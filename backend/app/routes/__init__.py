"""API route modules for the Guardian Check-In API."""

from typing import Optional

from fastapi import Depends, HTTPException

from app.auth import require_current_user
from app.models.auth import MeResponse
from app.routes.health import router as health_router
from app.routes.auth import router as auth_router
from app.routes.checkins import router as checkins_router
from app.routes.dashboard import router as dashboard_router
from app.routes.seniors import router as seniors_router
from app.routes.reports import router as reports_router
from app.routes.alerts import router as alerts_router
from app.routes.screenings import router as screenings_router
from app.routes.elevenlabs import router as elevenlabs_router


def register_routes(app) -> None:
    """Register all route modules with the FastAPI app."""
    app.include_router(health_router)
    app.include_router(auth_router, prefix="/auth", tags=["auth"])
    app.include_router(checkins_router, prefix="/checkins", tags=["checkins"])
    app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
    app.include_router(seniors_router, prefix="/seniors", tags=["seniors"])
    app.include_router(reports_router, prefix="/reports", tags=["reports"])
    app.include_router(alerts_router, prefix="/alerts", tags=["alerts"])
    app.include_router(screenings_router, prefix="/screenings", tags=["screenings"])
    app.include_router(elevenlabs_router, tags=["elevenlabs"])

    # Standalone /me endpoint at root level
    @app.get("/me", response_model=MeResponse)
    def me(user: Optional[dict] = Depends(require_current_user)) -> MeResponse:
        """Get current user information."""
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return MeResponse(
            email=user["email"],
            firstName=user.get("firstName", ""),
            lastName=user.get("lastName", ""),
            role=user.get("role", "senior"),
        )
