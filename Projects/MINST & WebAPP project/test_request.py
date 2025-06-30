import requests, os

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "received_input.png")

with open(IMAGE_PATH, "rb") as f:
    response = requests.post("http://127.0.0.1:5000/predict", files={"file": f})
    print(response.json())
