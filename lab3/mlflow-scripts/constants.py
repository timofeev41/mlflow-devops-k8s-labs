MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_HOST_HEADER = "mlflow.labs.itmo.loc"
EXPERIMENT_NAME = "Iris Classification Training"

REQUIREMENTS = "scikit-learn>=1.0.0\nmlflow>=2.0.0\npandas>=1.0.0\nnumpy>=1.0.0"

METRICS = ["accuracy", "precision", "recall", "f1_score"]

MODELS_FILES = {
    "logistic": {
        "metadata": "iris_logistic_regression_metadata.json",
        "model": "iris_logistic_regression.pkl",
    },
    "random": {
        "metadata": "iris_random_forest_metadata.json",
        "model": "iris_random_forest.pkl",
    },
}

MODELS_SUBDIR = "mlflow_models"
