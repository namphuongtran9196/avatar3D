import cv2
import json
import base64
import logging
import requests
import numpy as np

NGROK_API = "http://127.0.0.1:8000"
API_PYTORCH = f'{NGROK_API}/avatar3d/'
TOKEN = 'this_is_my_custom_token'
# USER_ID = "user_id"
USER_ID= [1,2,3,4]

def post_image(img_path):
    """ Encode image to base64 and send to server

    Args:
        img_path (str): path to image

    Returns:
        json: json response from server in bytes format
    """
    with open(img_path, 'rb') as f:
        # read image file in binary format
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    # add token to json data
    data = {"image": img_base64, "token": TOKEN, "user_id": USER_ID}
    # create a POST request
    response = requests.post(API_PYTORCH, data=json.dumps(data))
    return response.content

if __name__ == "__main__":
    img_path = "/home/kuhaku/Code/Buso/avatar3d/samples/nguyen.png"
    print(post_image(img_path))