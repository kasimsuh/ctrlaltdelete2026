"""Pydantic models for the Guardian Check-In API."""

from app.models.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    MeResponse,
    EphemeralTokenResponse,
)
from app.models.checkin import (
    TriageStatus,
    Answers,
    CheckinStartRequest,
    CheckinStartResponse,
    CheckinCompleteRequest,
    CheckinResult,
    CheckinUploadResponse,
    CheckinDetail,
    CheckinListResponse,
    FacialSymmetrySummary,
    FacialSymmetryResult,
    HeartRateResult,
)
from app.models.screening import (
    ScreeningResponseItem,
    ScreeningSession,
    ScreeningCreateRequest,
    ScreeningCreateResponse,
)
from app.models.report import (
    BaselineMetric,
    BaselineResponse,
    WeeklySummaryRequest,
    WeeklySummary,
    WeeklySummaryResponse,
    ReportOverview,
    ReportCheckinSummary,
    SeniorReportSummaryRequest,
    SeniorReportSummaryResponse,
)
from app.models.alert import (
    AlertLevel,
    AlertChannel,
    AlertRequest,
    AlertResponse,
)
from app.models.health import HealthStatus

__all__ = [
    # Auth
    "RegisterRequest",
    "LoginRequest",
    "TokenResponse",
    "MeResponse",
    "EphemeralTokenResponse",
    # Checkin
    "TriageStatus",
    "Answers",
    "CheckinStartRequest",
    "CheckinStartResponse",
    "CheckinCompleteRequest",
    "CheckinResult",
    "CheckinUploadResponse",
    "CheckinDetail",
    "CheckinListResponse",
    "FacialSymmetrySummary",
    "FacialSymmetryResult",
    "HeartRateResult",
    # Screening
    "ScreeningResponseItem",
    "ScreeningSession",
    "ScreeningCreateRequest",
    "ScreeningCreateResponse",
    # Report
    "BaselineMetric",
    "BaselineResponse",
    "WeeklySummaryRequest",
    "WeeklySummary",
    "WeeklySummaryResponse",
    "ReportOverview",
    "ReportCheckinSummary",
    "SeniorReportSummaryRequest",
    "SeniorReportSummaryResponse",
    # Alert
    "AlertLevel",
    "AlertChannel",
    "AlertRequest",
    "AlertResponse",
    # Health
    "HealthStatus",
]
