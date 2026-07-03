from sqlalchemy.orm import Session

from app.db import models
from app.schemas.student import StudentFeaturesIn


def create_student_with_prediction(db: Session, features: StudentFeaturesIn, prediction: dict) -> models.Student:
    student = models.Student(**features.model_dump())
    db.add(student)
    db.flush()  # assigns student.id without committing yet

    pred = models.Prediction(
        student_id=student.id,
        risk_category=prediction["risk_category"],
        prob_high_risk=prediction["prob_high_risk"],
        prob_moderate_risk=prediction["prob_moderate_risk"],
        prob_good_performance=prediction["prob_good_performance"],
        suggestions=prediction["suggestions"],
    )
    db.add(pred)
    db.commit()
    db.refresh(student)
    return student


def get_student(db: Session, student_id: int) -> models.Student | None:
    return db.query(models.Student).filter(models.Student.id == student_id).first()


def list_students(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Student).offset(skip).limit(limit).all()


def list_by_risk_category(db: Session, risk_category: str, skip: int = 0, limit: int = 100):
    return (
        db.query(models.Student)
        .join(models.Prediction)
        .filter(models.Prediction.risk_category == risk_category)
        .offset(skip)
        .limit(limit)
        .all()
    )
