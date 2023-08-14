import base64
import requests

image_filename='daisy.jpg'
with open(image_filename, "rb") as f:
    data = {"data": base64.b64encode(f.read()).decode("utf-8")}


r = requests.post('http://httpbin.org/post', json={"key": "value"})
print(data)
