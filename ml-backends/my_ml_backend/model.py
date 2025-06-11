from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import prediction
import utils
import numpy as np
import pandas as pd

class ip_predictor(prediction.BasePredictor):
        def __init__(self, label_group :str='breakdown_time', label_name :str= "放电时间", thredhold = 1000):
            super().__init__()
            self.label_group = label_group
            self.label_name = label_name
            self.thredhold = thredhold    

        def user_predict(self, task_data:pd.DataFrame):
            ip = np.array(task_data['ip'])
            time = task_data['time']
            mask = ip > self.thredhold
            diff = np.diff(mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            if mask[0]:
                starts = np.insert(starts,0,0)
            if mask[-1]:
                ends = np.append(ends, len(ip) - 1)
            target = np.argmax(ends-starts)
            return [prediction.Prediction(self.label_group, self.label_name, start=time[starts[target]], end=time[ends[target]])]



class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "test_script")
        self.predictor = ip_predictor()

    def get_data(self, tasks: List[Dict]):
        urls = {}
        for task in tasks:
            data = task['data']
            urls[data['shot']] = data['csv']
        data_dict = utils.load_data(urls)
        return data_dict

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')
        data_dict = self.get_data(tasks)
        model_preds = []
        for shot, data in data_dict.items():
            print(type(data))
            preds = prediction.convert_to_labelstudio_form(self.predictor.user_predict(data),'model_test')
            model_preds.append(preds)
        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]
        
        return ModelResponse(predictions=model_preds)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

