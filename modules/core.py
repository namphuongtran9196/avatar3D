import cv2
import base64
import json
import numpy as np

def decode_byte_img_base64(base_64_img):
    byte_img = base64.b64decode(base_64_img)
    jpg_as_np = np.frombuffer(byte_img, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    return img

def encode_byte_img_base64(img):
    data = {
        "image": base64.b64encode(img.tobytes()).decode('utf-8'),
        "shape": img.shape
    }
    return json.dumps(data)
