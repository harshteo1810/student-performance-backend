from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.student import StudentRecordOut
from app.core.security import verify_api_key
from app import crud

router = APIRouter(prefix="/students", tags=["students"], dependencies=[Depends(verify_api_key)])


@router.get("/", response_model=List[StudentRecordOut])
def list_students(
    skip: int = 0,
    limit: int = 100,
    risk_category: str | None = Query(None, description="Filter: High-Risk / Moderate-Risk / Good-Performance"),
    db: Session = Depends(get_db),
):
    if risk_category:
        return crud.list_by_risk_category(db, risk_category, skip, limit)
    return crud.list_students(db, skip, limit)


@router.get("/{student_id}", response_model=StudentRecordOut)
def get_student(student_id: int, db: Session = Depends(get_db)):
    student = crud.get_student(db, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student record not found")
    return student
