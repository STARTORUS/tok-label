from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from toklabel import utils, prediction
import requests
import os

model_url = os.getenv("MODEL_URL", "http://localhost:8000/")
model_url = "http://dap0.lan:30400/ml-models-discharge-timing-features/"

class NewModel(LabelStudioMLBase):
    """ ML Backend model for predicting Ip features
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.label_group = "breakdown_time"
        self.label_name = ["击穿时刻", "破裂时刻", "结束时刻"]

    def convert_predictions(self, predictions: list) -> list:
        """Convert predictions to label studio format
        """
        ls_results = []
        disrupted, ts, td, te = predictions
        for label_name, prediction in zip(self.label_name, [ts, td, te]):
            if label_name == "破裂时刻" and not disrupted:
                continue
            ls_results.append({
                "from_name": self.label_group,
                "to_name": "ts",
                "type": "timeserieslabels",
                "value": {
                    "start": prediction,
                    "end": prediction,
                    "instant": True,
                    "timeserieslabels": [label_name]
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
        
        # return all labels together in a single result
        ls_predictions = [{"result": ls_results}]
        
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

