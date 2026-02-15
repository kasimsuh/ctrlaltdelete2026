"""Health check Pydantic models."""

import ssl
import sys
from datetime import datetime
from typing import Optional

import pymongo
from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    status: str = Field(default="ok")
    time: datetime
    mongo: str = Field(default="unknown")
    mongo_host: Optional[str] = None
    mongo_db: Optional[str] = None
    mongo_error: Optional[str] = None
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    ssl: str = Field(default_factory=lambda: getattr(ssl, "OPENSSL_VERSION", "unknown"))
    pymongo: str = Field(
        default_factory=lambda: getattr(pymongo, "__version__", "unknown")
    )
    auth_required: bool = Field(default=False)
    jwt_configured: bool = Field(default=False)
