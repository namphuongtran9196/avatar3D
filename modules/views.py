import os 
from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt

import json
from .core import decode_byte_img_base64, encode_byte_img_base64

from modules.smplx.smplx_head.model import SMPLX_Head
from modules.smplx.smplx_pose.model import SMPLX_Pose
from modules.smplx.smplx_texture.model import SMPLX_Texture
from modules.smplx.smplx_uvmap.model import SMPLX_UVmap
    
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
    dir_name = 'test_upload'
    # For testing this, upload test images to `test_upload` folder
    # if there's no `test_upload` folder, please create folders `data/images` inside `smplx_pose`
    model.predict(dir_name, gender='male')

    model_head.predict(None)
    #model_pose.predict(None)
    #model_texture.predict(None)
    #model_uvmap.predict(None)
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
        jaw_pose, leye_pose, reye_pose, expression = data
        base_64_img_ret = encode_byte_img_base64(uvmap)
        if base_64_img_ret is None:
            return HttpResponse("Can not encode uvmap!")
        import numpy as np
        a = np.array([1,2,3])
        a.tolist()
        data = {"uvmap": base_64_img_ret, "jaw_pose": jaw_pose.tolist(), "leye_pose": leye_pose.tolist(), 
                        "reye_pose": reye_pose.tolist(), "expression": expression.tolist()}
        return HttpResponse(json.dumps(data))
