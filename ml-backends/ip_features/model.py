from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from toklabel import utils, prediction
import requests
import os


model_url = os.getenv("MODEL_URL", "http://localhost:8000/")
model_url = "http://dap0.lan:30400/ml-models-ip-features/"

class NewModel(LabelStudioMLBase):
    """ ML Backend model for predicting Ip features
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.label_group = "breakdown_time"
        self.label_name = "放电时间"

    def convert_predictions(self, predictions: list) -> list:
        """Convert predictions to label studio format
        """
        ls_results = []
        for p in predictions:
            ls_results.append({
                "from_name": self.label_group,
                "to_name": "ts",
                "type": "timeserieslabels",
                "value": {
                    "start": p[1],
                    "end": p[2],
                    "timeserieslabels": [self.label_name]
                }
            })
        return ls_results
    
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        # get model version
        model_version = requests.get(model_url + "version").json()["version"]
        # get shots
        shots = [task['data']['shot'] for task in tasks]
        # get predictions
        data = {"shot": shots}
        model_preds = requests.get(model_url + "predict", json=data).json()
        # convert to label studio format
        ls_results = self.convert_predictions(model_preds)
        ls_predictions = [{"result": [ls_result]} for ls_result in ls_results]
        # return model response
        return ModelResponse(model_version=model_version, predictions=ls_predictions)

    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        pass

