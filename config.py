from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Paths
    data_path: Path = Path("data/Student_Performance.csv")
    artifacts_dir: Path = Path("artifacts")
    model_path: Path = Path("artifacts/model.joblib")

    # Data
    target_col: str = "Performance Index"

    # Split
    test_size: float = 0.2
    random_state: int = 42

    # MLflow
    experiment_name: str = "student_performance_lasso"
