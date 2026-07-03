FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Model artifacts must exist before container start -- run train.py locally
# first (or as a build/release step) and make sure app/ml/artifacts/*.pkl
# and app/ml/student_performance_dataset.csv are committed or copied in.

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
