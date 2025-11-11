import mlflow
import time
import subprocess

def start_local_experiment( host='127.0.0.1',
                            port='8080',
                            uri=r'/mlruns',
                            experiment_name="ArticleRecommendator"
                            ):
    command = f'''mlflow server --host {host}  --port {port} \n
                mlflow ui --backend-store-uri {uri}'''
    print(command)

    result = subprocess.Popen(command, shell=True)

    mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

    mlflow.set_experiment(experiment_name)
