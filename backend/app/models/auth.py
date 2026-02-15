"""Authentication-related Pydantic models."""

from datetime import datetime

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    firstName: str
    lastName: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    email: str
    firstName: str = ""
    lastName: str = ""
    role: str = "senior"


class EphemeralTokenResponse(BaseModel):
    token: str
    expires_at: datetime
