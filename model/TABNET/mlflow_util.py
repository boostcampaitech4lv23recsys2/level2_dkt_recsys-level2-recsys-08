from pytorch_tabnet.callbacks import Callback
import mlflow

# MLflow class 선언
class MLCallback(Callback):
    def __init__(self, remote_server_uri, experiment_id, run_name, desc, params):
        self.remote_server_uri = remote_server_uri
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.desc = desc
        self.params = params
        self.best_auc = 0
        
    def on_train_begin(self, logs=None):
        mlflow.set_tracking_uri(self.remote_server_uri)
        mlflow.start_run(run_name = self.run_name, experiment_id=self.experiment_id)
        mlflow.log_params(self.params)
        
    def on_train_end(self, logs=None):
        
        mlflow.end_run()
        
    def on_epoch_end(self, epoch, logs=None):
    
        # send to MLFlow
        if self.best_auc < logs["valid_auc"]:
            self.best_auc = logs["valid_auc"]
            mlflow.log_metric("best_auc", self.best_auc)
        mlflow.log_metric("valid_auc", logs["valid_auc"])
        mlflow.log_metric("valid_ac", logs["valid_accuracy"])
        mlflow.log_metric("loss", logs["loss"])
        mlflow.log_metric("lr", logs["lr"])

def connect_server():
    remote_server_uri="http://118.67.134.110:30005"
    mlflow.set_tracking_uri(remote_server_uri)
    client = mlflow.tracking.MlflowClient()
    experiment_name = "TabNet" # 튜토
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    return remote_server_uri,experiment_id