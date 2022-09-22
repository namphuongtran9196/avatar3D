import json
import base64
import requests
import argparse
import time

NGROK_API = "http://127.0.0.1:8000"
API_PYTORCH = f'{NGROK_API}/avatar3d/'
TOKEN = 'this_is_my_custom_token'

def post_image(img_front_path,img_back_path,gender):
    """ Encode image to base64 and send to server

    Args:
        img_path (str): path to image

    Returns:
        json: json response from server in bytes format
    """
    with open(img_front_path, 'rb') as f:
        # read image file in binary format
        img_front_base64 = base64.b64encode(f.read()).decode('utf-8')
    with open(img_back_path, 'rb') as f:
        # read image file in binary format
        img_back_path = base64.b64encode(f.read()).decode('utf-8')
    # add token to json data
    data = {"image_front": img_front_base64,"image_back":img_back_path, "gender": gender, "token": TOKEN}
    # create a POST request
    response = requests.post(API_PYTORCH, data=json.dumps(data))
    return response.content

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--img_front', type=str, default='samples/005_front.jpg')
    parser.add_argument('-b', '--img_back', type=str, default='samples/005_back.jpg')
    parser.add_argument('-g', '--gender', type=str, default='neutral')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = args_parser()
    img_front_path = args.img_front
    img_back_path = args.img_back
    
    start = time.time()
    post_image(img_front_path,img_back_path, gender = args.gender)
    print(f'Elapsed time: {time.time() - start:.2f} seconds')