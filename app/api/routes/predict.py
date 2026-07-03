from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.student import StudentFeaturesIn, StudentRecordOut
from app.ml.inference import predict_risk
from app.ml.suggestions import generate_suggestions
from app.core.security import verify_api_key
from app import crud

router = APIRouter(prefix="/predict", tags=["predict"], dependencies=[Depends(verify_api_key)])


@router.post("/", response_model=StudentRecordOut)
def predict(features: StudentFeaturesIn, db: Session = Depends(get_db)):
    """
    Runs the trained model on submitted student features, generates
    personalized suggestions, persists both the input snapshot and the
    prediction (incl. suggestions), and returns the stored record.
    """
    feature_dict = features.model_dump(exclude={"external_student_id"})
    prediction = predict_risk(feature_dict)
    prediction["suggestions"] = generate_suggestions(feature_dict, prediction["risk_category"])
    student = crud.create_student_with_prediction(db, features, prediction)
    return student
