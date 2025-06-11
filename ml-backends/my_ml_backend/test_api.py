"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import pytest
import json
from model import NewModel


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=NewModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                # Your input test data here
                "csv": "http://dap0.lan:30422/files/ip_features/241021008.csv",
                "shot": 241021008
            }
        }],
        # Your labeling configuration here
        'label_config': '<View></View>'
    }

    expected_response = {
        'results': 
            # Your expected result here
            [
        {
          "type": "timeserieslabels",
          "value": {
            "end": 0.5680999999999999,
            "start": 0.5499999999999999,
            "timeserieslabels": [
              "放电时间"
            ]
          },
          "to_name": "ts",
          "from_name": "breakdown_time"
        }
      ]
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    print(response)
    assert response == expected_response
