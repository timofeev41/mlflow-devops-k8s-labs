mlflow server \
  --backend-store-uri sqlite:////var/lib/mlflow/db/mlflow.db \
  --default-artifact-root file:///var/lib/mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts mlflow.labs.itmo.loc,10.114.0.4,10.114.0.3,localhost,127.0.0.1 \
  --cors-allowed-origins http://mlflow.labs.itmo.loc,http://10.114.0.4,http://10.114.0.3



# С машины 10.114.0.3
curl -v http://10.114.0.4:5000 -H "Host: mlflow.labs.itmo.loc" 

-------------------------------------------------------------------

Все скрипты из папки mlflow-scripts запускаются на ВМ с mlflow в окружении mlflow-env