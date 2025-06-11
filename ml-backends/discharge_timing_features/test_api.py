"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""
# %%

import requests

url = "http://localhost:9090/predict"

request = {
    "tasks": [
        {
            "data": {"shot": "250310031"}
        }
    ]
}

result = requests.post(url, json=request)
print(result.json())

# %%
