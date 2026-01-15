#def main():
    #print("Hello from mlops-1!")
from src.train import train_regression
from src.evaluate import model_evaluate
import joblib
import mlflow
import mlflow.sklearn
import os


if __name__ == "__main__":

    #mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_tracking_uri("http://mlops-mlflow-server-1077737027.eu-west-2.elb.amazonaws.com")
  

    mlflow.set_experiment("student-performance")

    with mlflow.start_run(run_name="sgd_regression_rishika"):
        model, scaler, X_test, y_test = train_regression()

        print("----------Gradient Descent Model Trained and Saved Successfully------------")

        #save trained model
        """
        joblib.dump(
        {"model": model, "scaler": scaler},
        "models/student_performance_sgd.joblib"
        )"""

        os.makedirs("models", exist_ok=True)

        joblib.dump(
        {"model": model, "scaler": scaler},
        "models/student_performance_sgd.joblib"
        )


        #mlflow.log_artifact("models/student_performance_sgd.joblib", artifact_path="model")

        # Log & register model (auto-versioned)
        mlflow.sklearn.log_model(
            model,
            name='model',
            registered_model_name='sgd_student_performance'
        )

        metrics = model_evaluate(model, X_test, y_test)
        print("\n------------Model Metrics------------")
        print(metrics)


        mlflow.log_metric("mse", metrics["mse"])
        mlflow.log_metric("r2", metrics["r2"])



        mlflow.log_param("model_type", "SGDRegressor")
        mlflow.log_param("loss", "squared_error")
        mlflow.log_param("learning_rate", "constant")
        mlflow.log_param("eta0", 0.01)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        
        
