import os 
from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt

import json
from .core import decode_byte_img_base64, encode_byte_img_base64

from modules.smplx.smplx_head.model import SMPLX_Head
from modules.smplx.smplx_pose.model import SMPLX_Pose
from modules.smplx.smplx_texture.model import SMPLX_Texture
from modules.smplx.smplx_uvmap.model import SMPLX_UVmap
import numpy as np
# Init models
model_head = SMPLX_Head(
                        model_3dffa_config_path=os.path.abspath("modules/smplx/smplx_head/src/V2_3DDFA/configs/mb1_120x120.yml"), 
                        model_smplx_path=os.path.abspath("modules/smplx/smplx_head/models/smplx"),
                        max_iter=10000,
                        uvmap_path = os.path.abspath("modules/smplx/smplx_head/src/uvmap.obj")
                        )

model_pose = SMPLX_Pose()
model_texture = SMPLX_Texture()
model_uvmap = SMPLX_UVmap()
    
def index(request):
    model_head.predict(None)
    model_pose.predict(None)
    model_texture.predict(None)
    model_uvmap.predict(None)
    return HttpResponse("Test model finished! Please check the console output.")

@csrf_exempt
def avatar3d(request):
    if request.method == 'GET':
        return HttpResponse("Hello, GET. You're at the tensorflow_model index. \n")
    elif request.method == 'POST':
        # Get data from request
        byte_data = request.body
        json_data = byte_data.decode('utf8').replace("'", '"')
        json_data = json.loads(json_data)

        # Checking the token
        if json_data.get("token", "") != "this_is_my_custom_token":
            return HttpResponse("Invalid token!")

        # Decode image from base64
        base_64_img = json_data["image"]
        img = decode_byte_img_base64(base_64_img)

        uvmap, data = model_head.predict(img)

        vt = [[1,2,3],[3,4,5]]
        vn = [[1,2,3],[3,4,5]]
        fv = [[1,2,3],[3,4,5]]
        ft = [[1,2,3],[3,4,5]]
        fn = [[1,2,3],[3,4,5]]

        base_64_img_ret = encode_byte_img_base64(uvmap)
        if base_64_img_ret is None:
            return HttpResponse("Can not encode uvmap!")
        data = {"uvmap": base_64_img_ret, "vt": vt, "vn": vn, "fv": fv, "ft": ft, "fn": fn}
        return HttpResponse(json.dumps(data))