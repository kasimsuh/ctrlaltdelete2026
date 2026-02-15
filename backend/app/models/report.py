"""Report-related Pydantic models."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BaselineMetric(BaseModel):
    metric_type: str
    sample_count: int
    mean: float
    std_dev: float
    last_value: float
    updated_at: datetime


class BaselineResponse(BaseModel):
    senior_id: str
    metrics: List[BaselineMetric]


class WeeklySummaryRequest(BaseModel):
    week_start: Optional[str] = None


class WeeklySummary(BaseModel):
    summary_id: str
    senior_id: str
    week_start: str
    week_end: str
    summary_text: str
    key_trends: List[str]
    created_at: datetime


class WeeklySummaryResponse(BaseModel):
    senior_id: str
    summary: WeeklySummary


class ReportOverview(BaseModel):
    total_checkins: int
    last_checkin_at: Optional[str] = None
    days_since_last_checkin: Optional[int] = None
    triage_counts: Dict[str, int] = Field(default_factory=dict)
    signal_counts: Dict[str, int] = Field(default_factory=dict)


class ReportCheckinSummary(BaseModel):
    completed_at: Optional[str] = None
    triage_status: Optional[str] = None
    triage_reasons: List[str] = []


class SeniorReportSummaryRequest(BaseModel):
    senior_id: str
    senior_name: Optional[str] = None
    senior_email: Optional[str] = None
    overview: ReportOverview
    recent_checkins: List[ReportCheckinSummary] = Field(default_factory=list)


class SeniorReportSummaryResponse(BaseModel):
    summary: str
    symptoms: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    follow_up: List[str] = Field(default_factory=list)
