"""Authentication routes."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from google import genai

from app.auth import (
    require_current_user,
    create_user,
    authenticate_user,
    create_access_token,
)
from app.models.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    MeResponse,
    EphemeralTokenResponse,
)

router = APIRouter()
logger = logging.getLogger("guardian")


@router.post("/register", response_model=MeResponse)
def register(payload: RegisterRequest) -> MeResponse:
    """Register a new user account."""
    user = create_user(
        email=payload.email.strip().lower(),
        password=payload.password,
        firstName=payload.firstName,
        lastName=payload.lastName,
    )
    return MeResponse(
        email=user["email"],
        firstName=user.get("firstName", ""),
        lastName=user.get("lastName", ""),
        role=user.get("role", "senior"),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    """Authenticate user and return access token."""
    user = authenticate_user(
        email=payload.email.strip().lower(), password=payload.password
    )
    token = create_access_token(sub=str(user["_id"]), email=user["email"])
    return TokenResponse(access_token=token)


@router.post("/ephemeral", response_model=EphemeralTokenResponse)
@router.get("/ephemeral", response_model=EphemeralTokenResponse)
def create_ephemeral_token() -> EphemeralTokenResponse:
    """Create a short-lived Gemini API token for frontend use."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key missing for /auth/ephemeral")
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set (set it in your shell env or in the repo root .env)",
        )

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"},
    )

    now = datetime.now(timezone.utc)
    try:
        token = client.auth_tokens.create(
            config={
                "uses": 1,
                "expire_time": now + timedelta(minutes=30),
                "new_session_expire_time": now + timedelta(minutes=5),
            }
        )
    except Exception as exc:
        logger.exception("Failed to create Gemini ephemeral token")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to create Gemini ephemeral token: {exc.__class__.__name__}",
        ) from exc

    logger.info("Gemini ephemeral token created")
    return EphemeralTokenResponse(
        token=token.name, expires_at=now + timedelta(minutes=30)
    )
