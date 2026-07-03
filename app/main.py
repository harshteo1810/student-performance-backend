from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import engine, Base
from app.api.routes import predict, students

# Create tables on startup if they don't exist (fine for SQLite/dev;
# for Postgres in production you'd typically use Alembic migrations instead)
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME)

# CORS: open for now since the frontend (Streamlit) runs on a different
# origin/port. Tighten allow_origins to your actual frontend URL before
# going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix=settings.API_V1_PREFIX)
app.include_router(students.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def health_check():
    return {"status": "ok", "service": settings.APP_NAME}
