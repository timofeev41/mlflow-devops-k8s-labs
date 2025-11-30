import os
import warnings
import urllib3
import json
import tempfile

print("="*80)
print("ДОБАВЛЕНИЕ АРТЕФАКТОВ IN RUNS")
print("="*80)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

from constants import (
    MLFLOW_TRACKING_URI,
    MLFLOW_HOST_HEADER,
    EXPERIMENT_NAME,
    REQUIREMENTS,
    METRICS,
    MODELS_FILES,
    MODELS_SUBDIR,
)

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = "true"

if 'MLFLOW_TRACKING_SERVER_CERT_PATH' in os.environ:
    del os.environ['MLFLOW_TRACKING_SERVER_CERT_PATH']

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

print(f"\n[КОНФИГУРАЦИЯ]")
print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"  - Host Header: {MLFLOW_HOST_HEADER}")
print(f"  - Эксперимент: {EXPERIMENT_NAME}")

import requests

original_session_init = requests.Session.__init__

def patched_session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    self.headers.update({'Host': MLFLOW_HOST_HEADER})

requests.Session.__init__ = patched_session_init

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


print("\n" + "="*80)
print("ПОЛУЧЕНИЕ СПИСКА RUNS")
print("="*80)

try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"✗ Эксперимент не найден: {EXPERIMENT_NAME}")
        exit(1)

    print(f"\n✓ Эксперимент найден: {experiment.name} (ID: {experiment.experiment_id})")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"✓ Найдено runs: {len(runs)}")

    for i, run in enumerate(runs, 1):
        print(f"  {i}. {run.info.run_name} (ID: {run.info.run_id})")

except Exception as e:
    print(f"\n✗ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


print("\n" + "="*80)
print("ДОБАВЛЕНИЕ АРТЕФАКТОВ")
print("="*80)

MODELS_DIR = os.path.join(tempfile.gettempdir(), MODELS_SUBDIR)

mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

for run in runs:
    print(f"\n[{run.info.run_name}]")

    try:
        model_name = run.info.run_name.lower()

        # find model type by substring and map to files via MODELS_FILES
        matched = False
        for key, files in MODELS_FILES.items():
            if key in model_name:
                metadata_file = os.path.join(MODELS_DIR, files["metadata"])
                model_file = os.path.join(MODELS_DIR, files["model"])
                matched = True
                break

        if not matched:
            continue

        if not os.path.exists(metadata_file):
            print(f"  ⚠ Метаданные не найдены: {metadata_file}")
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        with mlflow.start_run(run_id=run.info.run_id):

            model_info = {
                "model_type": metadata["model_type"],
                "framework": metadata["framework"],
                "training_date": metadata["timestamp"],
                "dataset_info": {
                    "train_size": metadata["train_size"],
                    "test_size": metadata["test_size"],
                    "dataset_name": metadata["dataset"]
                }
            }

            temp_file = os.path.join(tempfile.gettempdir(), f"model_info_{run.info.run_id}.json")
            with open(temp_file, 'w') as f:
                json.dump(model_info, f, indent=2)

            mlflow.log_artifact(temp_file, artifact_path="model_info")
            print(f"  ✓ model_info.json логирован")

            req_file = os.path.join(tempfile.gettempdir(), f"requirements_{run.info.run_id}.txt")
            requirements = REQUIREMENTS
            with open(req_file, 'w') as f:
                f.write(requirements)

            mlflow.log_artifact(req_file, artifact_path="requirements")
            print(f"  ✓ requirements.txt логирован")

            import csv
            metrics_data = {
                "metric": METRICS,
                "value": [metadata["metrics"].get(m) for m in METRICS],
            }

            csv_file = os.path.join(tempfile.gettempdir(), f"metrics_{run.info.run_id}.csv")
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(metrics_data["metric"])
                writer.writerow(metrics_data["value"])

            mlflow.log_artifact(csv_file, artifact_path="metrics")
            print(f"  ✓ metrics.csv логирован")

            config_file = os.path.join(tempfile.gettempdir(), f"config_{run.info.run_id}.json")
            with open(config_file, 'w') as f:
                json.dump(metadata["parameters"], f, indent=2)

            mlflow.log_artifact(config_file, artifact_path="model_config")
            print(f"  ✓ model_config.json логирован")

            if os.path.exists(model_file):
                import shutil
                temp_model = os.path.join(tempfile.gettempdir(), f"model_{run.info.run_id}.pkl")
                shutil.copy(model_file, temp_model)

                mlflow.log_artifact(temp_model, artifact_path="model")
                print(f"  ✓ Модель (pickle) логирована")

    except Exception as e:
        print(f"  ✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


print("\n" + "="*80)
print("ПРОВЕРКА АРТЕФАКТОВ")
print("="*80)

try:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        print(f"\n[{run.info.run_name}]")

        if not artifacts:
            print(f"  - Артефактов нет")
        else:
            print(f"  - Артефактов: {len(artifacts)}")
            for artifact in artifacts:
                print(f"    - {artifact.path}")

except Exception as e:
    print(f"\n✗ ОШИБКА при проверке: {e}")
