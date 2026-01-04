# src/train.py

import os
import joblib

def train_model(
    model,
    X_train,
    y_train,
    scaler,
    epochs,
    batch_size,
    validation_split,
    artifacts_dir="artifacts"
):
    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # ---------------------------
    # SAVE MODEL & SCALER
    # ---------------------------
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "model.h5") 
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    return history
