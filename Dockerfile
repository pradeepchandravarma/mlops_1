FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY src/api.py .
COPY src/app_streamlit.py .
COPY src/train_rf_mlflow.py .
COPY src/register_best_model.py .
COPY src/export_model.py .

# ---- Entrypoint (NOTE: from utils/) ----
COPY utils/entrypoint.sh .

# ---- Data ----
COPY data/ ./data/

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8501

CMD ["/app/entrypoint.sh"]