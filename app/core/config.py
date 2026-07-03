import os

from fastapi import Header, HTTPException


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./student_performance.db")
    APP_NAME: str = "Student Performance Evaluation API"
    API_V1_PREFIX: str = "/api/v1"


settings = Settings()

API_KEY = os.getenv("API_KEY", "")  # set this in Render's env vars


def verify_api_key(x_api_key: str = Header(default="")):
    """
    Lightweight protection -- not real auth, just stops random internet
    traffic from hitting the free-tier instance. If API_KEY isn't set
    (e.g. local dev), this check is skipped entirely.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
