from src.config import *
from src.data_loader import load_data
from src.preprocessing import encode_features, split_data, scale_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict

def main():
    df = load_data(DATA_PATH)
    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(
        df, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model = build_model(X_train_scaled.shape[1])

    history = train_model(
           model=model,
           X_train=X_train_scaled,
           y_train=y_train,
           X_test=X_test_scaled,
           y_test=y_test,
           scaler=scaler,
           epochs=EPOCHS,
           batch_size=BATCH_SIZE,
           validation_split=VALIDATION_SPLIT
    )

    metrics = evaluate_model(model, X_test_scaled, y_test)
    print(metrics)

    new_data = [[6, 80, 1, 7, 3]]
    prediction = predict(model, scaler, new_data)
    print("Predicted Performance Index:", prediction)

if __name__ == "__main__":
    main()
