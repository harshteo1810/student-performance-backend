# Student Performance Evaluation — Backend

FastAPI service that serves a trained RandomForest classifier predicting
student risk category (`High-Risk` / `Moderate-Risk` / `Good-Performance`)
from academic, attendance, study-hours, behavioral, and internal-assessment
features, and persists every prediction to a database.

## Project structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app + router registration
│   ├── core/config.py           # settings (DATABASE_URL, etc.)
│   ├── db/
│   │   ├── database.py          # SQLAlchemy engine/session
│   │   └── models.py            # Student, Prediction tables
│   ├── schemas/student.py       # Pydantic request/response models
│   ├── ml/
│   │   ├── train.py             # trains model from CSV, saves artifacts
│   │   ├── inference.py         # loads artifacts, runs predictions
│   │   ├── student_performance_dataset.csv
│   │   └── artifacts/           # model.pkl, encoders, metrics.json
│   ├── api/routes/
│   │   ├── predict.py           # POST /api/v1/predict/
│   │   └── students.py          # GET  /api/v1/students/, /{id}
│   └── crud.py
├── requirements.txt
├── Dockerfile
└── render.yaml
```

## Local setup

```bash
cd backend
pip install -r requirements.txt

# Train the model (only needed once, or whenever the dataset changes)
python3 -m app.ml.train

# Run the API
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: `http://localhost:8000/docs` (FastAPI auto-generates
Swagger UI — use this to test endpoints manually before wiring up the
Streamlit frontend).

## Endpoints

### `POST /api/v1/predict/`
Submit a student's features, get back a stored prediction.

Request body (see `app/schemas/student.py` for full validation ranges):
```json
{
  "external_student_id": "ABES2024001",
  "age": 20,
  "gender": "Male",
  "years_in_program": 3,
  "cumulative_gpa": 5.2,
  "backlogs_count": 3,
  "attendance_percentage": 58,
  "consecutive_absences_max": 6,
  "daily_study_hours": 1.2,
  "class_participation_score": 3.5,
  "extracurricular_involvement_score": 4.0,
  "peer_collaboration_score": 4.5,
  "ct1_marks_25": 12,
  "ct2_marks_25": 10,
  "assignment_marks_15": 8,
  "mid_sem_marks_35": 15
}
```

Response:
```json
{
  "id": 1,
  "external_student_id": "ABES2024001",
  "created_at": "2026-07-02T04:38:59",
  "prediction": {
    "risk_category": "High-Risk",
    "prob_high_risk": 0.975,
    "prob_moderate_risk": 0.025,
    "prob_good_performance": 0.0
  }
}
```

### `GET /api/v1/students/`
List all evaluated students. Query params: `skip`, `limit`,
`risk_category` (filter by `High-Risk` / `Moderate-Risk` / `Good-Performance`).

### `GET /api/v1/students/{id}`
Fetch a single student record + its prediction.

## Retraining

Whenever the dataset changes, re-run `python3 -m app.ml.train` — this
overwrites `app/ml/artifacts/*.pkl`. The API loads artifacts once at
import time (`app/ml/inference.py`), so **restart the server** after
retraining for changes to take effect.

`train.py` also writes `metrics.json` with the full classification report
and macro F1 — pull numbers from there for your report rather than
re-deriving them.

## Deployment (Render)

1. Push this `backend/` folder to a GitHub repo (as its own repo, or as a
   subdirectory — Render lets you set a root directory).
2. On Render: New → Web Service → connect the repo → Render auto-detects
   `render.yaml`, or set manually: Environment = Docker, root dir = `backend`.
3. **Important — model artifacts:** `app/ml/artifacts/*.pkl` must exist in
   the image. Either commit them to git (simplest — they're small, a few
   MB) or add a Render "build command" that runs `python3 -m app.ml.train`
   before start.
4. **Important — SQLite persistence:** Render's free-tier filesystem is
   ephemeral and resets on every redeploy, so SQLite data (students/
   predictions you've logged) will be wiped each time you push a new
   version. Fine for development/demo. For a persistent deployment, add
   Render's free Postgres add-on and set `DATABASE_URL` to the provided
   Postgres connection string — no code changes needed, SQLAlchemy handles
   both.

## Frontend integration (Streamlit)

CORS is currently open (`allow_origins=["*"]` in `main.py`) so a Streamlit
app calling this API from any origin will work out of the box. Tighten
this to your actual Streamlit deployment URL before considering this
production-ready.

Example call from Streamlit:
```python
import requests
resp = requests.post("https://your-api.onrender.com/api/v1/predict/", json=student_data)
result = resp.json()
```

## API Key Protection

`/predict` and `/students` endpoints require an `X-API-Key` header matching
the `API_KEY` environment variable. This is lightweight protection (a
single shared secret), not real per-user auth -- sufficient for a college
project to keep the free-tier instance from being hit by random traffic,
not a substitute for proper authentication if this ever serves multiple
real users.

**Local dev:** if `API_KEY` isn't set, the check is skipped entirely (open
access) so you don't need to configure anything to test locally.

**On Render:** set `API_KEY` in the web service's Environment tab (same
place you added `DATABASE_URL`) to any secret string of your choice.
After that, every request needs the header:
```
X-API-Key: <your-secret>
```
In Swagger UI (`/docs`), click the padlock icon (or "Authorize" button)
and enter your key there once, and it'll apply to all requests you test
from that page.

## Known simplifications (documented, not accidental)

- **Auth:** none yet. Anyone with the URL can POST predictions or read
  student records. Fine for a coursework demo; add JWT-based auth
  (e.g. `fastapi-users` or a hand-rolled OAuth2PasswordBearer flow) before
  any real deployment with real student data.
- **Single-row missing-value imputation:** `inference.py` fills a missing
  `extracurricular_involvement_score` with a fixed 5.0 (scale midpoint)
  rather than the training-set median, since the median isn't saved as an
  artifact. Fine for single-row API calls; if you add batch prediction,
  save and reuse the training median instead.
