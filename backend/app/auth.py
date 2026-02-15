from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo.errors import PyMongoError

from app.db import get_mongo_client


# pbkdf2_sha256 is available without platform-specific wheels, which keeps setup
# simple for hackathon environments.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def _jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        # Explicit error to avoid silently issuing unverifiable tokens.
        raise RuntimeError("JWT_SECRET is not set")
    return secret


def _jwt_alg() -> str:
    return os.environ.get("JWT_ALG", "HS256")


def _jwt_expires_minutes() -> int:
    try:
        return int(os.environ.get("JWT_EXPIRES_MIN", "60"))
    except ValueError:
        return 60


def _users_collection():
    client = get_mongo_client()
    # Use the database specified in the URI path. If the URI has no database part,
    # `get_default_database()` raises; fall back to env/default.
    try:
        db = client.get_default_database()
    except Exception:
        db = None
    if db is None:
        db_name = os.environ.get("MONGO_DB", "guardian")
        db = client[db_name]
    return db["users"]

def ensure_user_indexes() -> None:
    users = _users_collection()
    # Unique email; safe to call repeatedly.
    try:
        users.create_index("email", unique=True)
    except Exception:
        # Index creation can fail on limited perms or races; auth still works with best-effort uniqueness.
        pass


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(*, sub: str, email: str) -> str:
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(minutes=_jwt_expires_minutes())
    payload = {
        "sub": sub,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=_jwt_alg())


def get_user_by_email(email: str) -> Optional[dict]:
    return _users_collection().find_one({"email": email})


def create_user(*, email: str, password: str, firstName: str, lastName: str) -> dict:
    users = _users_collection()
    try:
        existing = users.find_one({"email": email})
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e.__class__.__name__}")
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    doc = {
        "email": email,
        "password_hash": hash_password(password),
        "firstName": firstName,
        "lastName": lastName,
        "role": "senior",
        "created_at": datetime.utcnow(),
    }
    try:
        res = users.insert_one(doc)
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e.__class__.__name__}")
    doc["_id"] = res.inserted_id
    return doc


def authenticate_user(email: str, password: str) -> dict:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user


def require_auth_enabled() -> bool:
    return os.environ.get("REQUIRE_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}


def require_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[dict]:
    """
    If REQUIRE_AUTH=true: missing/invalid token -> 401.
    If REQUIRE_AUTH=false: returns None when missing token.
    """
    if creds is None:
        if require_auth_enabled():
            raise HTTPException(status_code=401, detail="Missing bearer token")
        return None

    token = creds.credentials
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[_jwt_alg()])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    email = payload.get("email")
    sub = payload.get("sub")
    if not email or not sub:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
