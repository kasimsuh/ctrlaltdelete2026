"""Alert-related Pydantic models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class AlertLevel(str, Enum):
    YELLOW = "yellow"
    RED = "red"


class AlertChannel(str, Enum):
    SMS = "sms"
    VOICE = "voice"
    EMAIL = "email"


class AlertRequest(BaseModel):
    senior_id: str
    level: AlertLevel
    channel: AlertChannel
    target: str
    message: Optional[str] = None


class AlertResponse(BaseModel):
    alert_id: str
    senior_id: str
    level: AlertLevel
    channel: AlertChannel
    target: str
    status: str
    created_at: datetime
