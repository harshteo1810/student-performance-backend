from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class StudentFeaturesIn(BaseModel):
    """Input schema for a prediction request -- must match training features exactly."""
    external_student_id: Optional[str] = Field(None, description="College roll no. or similar")
    age: int = Field(..., ge=15, le=30)
    gender: str = Field(..., description="Male / Female / Other")
    years_in_program: int = Field(..., ge=1, le=6)
    cumulative_gpa: float = Field(..., ge=0, le=10)
    backlogs_count: int = Field(..., ge=0)
    attendance_percentage: float = Field(..., ge=0, le=100)
    consecutive_absences_max: int = Field(..., ge=0)
    daily_study_hours: float = Field(..., ge=0, le=24)
    class_participation_score: float = Field(..., ge=0, le=10)
    extracurricular_involvement_score: Optional[float] = Field(None, ge=0, le=10)
    peer_collaboration_score: float = Field(..., ge=0, le=10)
    ct1_marks_25: float = Field(..., ge=0, le=25)
    ct2_marks_25: float = Field(..., ge=0, le=25)
    assignment_marks_15: float = Field(..., ge=0, le=15)
    mid_sem_marks_35: float = Field(..., ge=0, le=35)


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    student_id: int
    risk_category: str
    prob_high_risk: float
    prob_moderate_risk: float
    prob_good_performance: float
    suggestions: list[str] = []
    created_at: datetime


class StudentRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_student_id: Optional[str]
    created_at: datetime
    prediction: Optional[PredictionOut] = None
