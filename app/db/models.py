from datetime import datetime

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.db.database import Base


class Student(Base):
    """
    Stores the raw feature snapshot submitted for a prediction.
    One student can have multiple snapshots over time (e.g. re-evaluated
    each month), so this is intentionally not a strict 1-row-per-student
    table -- each row is one evaluation instance.
    """
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    external_student_id = Column(String, index=True, nullable=True)  # e.g. college roll no.
    age = Column(Integer)
    gender = Column(String)
    years_in_program = Column(Integer)
    cumulative_gpa = Column(Float)
    backlogs_count = Column(Integer)
    attendance_percentage = Column(Float)
    consecutive_absences_max = Column(Integer)
    daily_study_hours = Column(Float)
    class_participation_score = Column(Float)
    extracurricular_involvement_score = Column(Float, nullable=True)
    peer_collaboration_score = Column(Float)
    ct1_marks_25 = Column(Float)
    ct2_marks_25 = Column(Float)
    assignment_marks_15 = Column(Float)
    mid_sem_marks_35 = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="student", uselist=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), unique=True)
    risk_category = Column(String, index=True)
    prob_high_risk = Column(Float)
    prob_moderate_risk = Column(Float)
    prob_good_performance = Column(Float)
    suggestions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="prediction")
