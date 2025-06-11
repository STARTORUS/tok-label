from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from toklabel import utils, prediction, maskutils
from toklabel.maskutils import CAMERA_ENV
import sunist2.script.postgres as pg
import numpy as np
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.getenv("MODEL_PATH", os.path.join(dir_path, "model.pth"))
PARAS = {'center':(0.5, 0), 
         'a': 0.3, 
         'k':1.75, 
         'delta_u':0.3125,
         'delta_l':-0.25}

class LCFSModel(LabelStudioMLBase):
    """ 
    ML Backend model for predicting Ip features
    """

    def setup(self):
        """
        Configure any parameters of your model here
        """
        self.set("model_version", "plasma_mask_script")
        self.height = CAMERA_ENV['image_param']['h_camera']
        self.weight = CAMERA_ENV['image_param']['w_camera']

        
    def read_context(self, context: Dict):
        param = {}
        num_pred_list = []
        center_r = None
        center_z = None
        for num in context['result']:
            if num.get('type',None) == 'number':
                para_name = num['from_name']
                para_value = num['value']['number']
                print(f'name={para_name}')
                print(f'value={para_value}')
                if para_name == 'R':
                    center_r = float(para_value)/100
                    continue
                if para_name == 'Z':
                    center_z = float(para_value)/100
                    continue
                if para_name == 'a':
                    para_value = float(para_value)/100
                param[para_name] = para_value
                num_pred_list.append(prediction.Number(para_value, para_name))
        center_r =PARAS['center'][0] if not center_r else center_r       
        center_z =PARAS['center'][1] if not center_z else center_z       
        param['center'] = (center_r, center_z)
        return param, num_pred_list

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        param = PARAS
        Pred_list = []
        # 获取相机环境参数
        shot = tasks[0]['data']['shot']
        conn = pg.connect_to_database()
        env_camera = pg.load_env(conn, shot, pg.ENV_CAMERA)
        pg.close_connection(conn)

        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
            print('No context')
            #print(tasks[0]['annotations'])
            #if tasks[0]['annotations']:
            #    param, Pred_list = self.read_context(tasks[0]['annotations'][0])
            return ModelResponse(predictions=[])
        else:
            print('get Context')
            print(context)
            param_con, Pred_list = self.read_context(context)
            param.update(param_con)
            #param, Pred_list = self.read_context(tasks[0]['annotations'][0])
        if len(param) < 5:
            return ModelResponse(predictions=[])
        poly_l, poly_r = maskutils.generate_plasma_polygon(**param, env=env_camera, angle=(0.5969, 2.7960))
        center_l, center_r = maskutils.generate_plasma_center(param['center'], env=env_camera)
        Pred_list = []
        Pred_list.append(prediction.Polygon(maskutils.pixel_to_percent(poly_l),'left','plasma'))
        Pred_list.append(prediction.Polygon(maskutils.pixel_to_percent(poly_r),'right','plasma'))
        Pred_list.append(prediction.KeyPoint(center_l, 'left', 'center', point_width=0.25))
        Pred_list.append(prediction.KeyPoint(center_r, 'right', 'center', point_width=0.25))
        pred_res = prediction.to_labelstudio_form(Pred_list, model_version='plasma_polygon_script')
        return ModelResponse(predictions=pred_res)



    # def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
    #     """ Write your inference logic here
    #         :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
    #         :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
    #         :return model_response
    #             ModelResponse(predictions=predictions) with
    #             predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
    #     """
    #     print(f'''\
    #     Run prediction on {tasks}
    #     Received context: {context}
    #     Project ID: {self.project_id}
    #     Label config: {self.label_config}
    #     Parsed JSON Label config: {self.parsed_label_config}
    #     Extra params: {self.extra_params}''')
    #     data_dict = self.get_data(tasks)
    #     model_preds = []
    #     for shot, data in data_dict.items():
    #         preds = prediction.convert_to_labelstudio_form(self.ip_predictor.predict(data), self.model_version)
    #         model_preds.append(preds)
    #     # example for resource downloading from Label Studio instance,
    #     # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
    #     # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

    #     # example for simple classification
    #     # return [{
    #     #     "model_version": self.get("model_version"),
    #     #     "score": 0.12,
    #     #     "result": [{
    #     #         "id": "vgzE336-a8",
    #     #         "from_name": "sentiment",
    #     #         "to_name": "text",
    #     #         "type": "choices",
    #     #         "value": {
    #     #             "choices": [ "Negative" ]
    #     #         }
    #     #     }]
    #     # }]
        
    #     return ModelResponse(predictions=model_preds)
    
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
