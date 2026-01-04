#!/bin/sh
set -e

case "$APP_MODE" in
  train) python train_rf_mlflow.py ;;
  register) python register_best_model.py ;;
  export) python export_model.py ;;
  api) uvicorn api:app --host 0.0.0.0 --port 8000 ;;
  ui) streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0 ;;
  *) echo "Invalid APP_MODE"; exit 1 ;;
esac