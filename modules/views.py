import os
import cv2
import torch
import numpy as np
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

import json
from .core import decode_byte_img_base64, encode_byte_img_base64

from modules.smplx.smplx_head.model import SMPLX_Head
from modules.smplx.smplx_pose.model import SMPLX_Pose
from modules.smplx.smplx_texture.model import SMPLX_Texture
from modules.smplx.smplx_uvmap.model import SMPLX_UVmap
from human_body_prior.tools.model_loader import load_vposer

# Init models
# model_head = SMPLX_Head(
#                         model_3dffa_config_path=os.path.abspath("modules/smplx/smplx_head/src/V2_3DDFA/configs/mb1_120x120.yml"), 
#                         model_smplx_path=os.path.abspath("modules/smplx/smplx_pose/smplifyx/models/smplx"),
#                         max_iter=10000,
#                         uvmap_path = os.path.abspath("modules/smplx/smplx_head/src/uvmap.obj"),
#                         num_expression_coeffs=10,
#                         )

model_pose = SMPLX_Pose()
# model_texture = SMPLX_Texture()
# model_uvmap = SMPLX_UVmap()

# # For decode the body pose
# vposer, _ = load_vposer(os.path.abspath("modules/smplx/smplx_pose/smplifyx/models/vposer_v1_0"), vp_model='snapshot')
# vposer = vposer.to(device='cpu')
# vposer.eval()

# def index(request):
#     dir_name = 'test_upload'
#     smplifyx_out_path = '/Users/jaydentran1909/Documents/avatar3d/modules/smplx/smplx_pose/data/smplifyx_results'
#     front_img_path = '/Users/jaydentran1909/Documents/avatar3d/modules/smplx/smplx_pose/data/images/005_front.jpg'
#     back_img_path = '/Users/jaydentran1909/Documents/avatar3d/modules/smplx/smplx_pose/data/images/005_back.jpg'
#     model.predict(smplifyx_out_path, front_img_path, back_img_path, dir_name)
#     return HttpResponse("Test model finished! Please check the console output.")

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

        gender = json_data["gender"]
        # Decode image from base64
        base_64_img_front = json_data["image_front"]
        base_64_img_back = json_data["image_back"]
        img_front = decode_byte_img_base64(base_64_img_front)
        img_back = decode_byte_img_base64(base_64_img_back)

        # Write image to file for running openpose using cmd
        cv2.imwrite("modules/smplx/smplx_pose/data/images/inputs/front.jpg", img_front)
        cv2.imwrite("modules/smplx/smplx_pose/data/images/inputs/back.jpg", img_back)

        # # Predict the smplx parameters
        result = model_pose.predict("inputs", gender)
        
        # # Predict the smplx head based on the smplx parameters
        # model_head.set_face_smplify(result['front'][0]['result']['jaw_pose'], 
        #                             result['front'][0]['result']['leye_pose'], 
        #                             result['front'][0]['result']['reye_pose'], 
        #                             result['front'][0]['result']['expression'])
        
        # uvmap_face, smplx_face_params, smplx_uv_params = model_head.predict(img_front, gender=gender)
        
        
        # # SMPLX parameters
        # betas = torch.from_numpy(result['front'][0]['result']['betas']).float()
        # left_hand_pose = torch.from_numpy(result['front'][0]['result']['left_hand_pose']).float() # not fix yet
        # right_hand_pose = torch.from_numpy(result['front'][0]['result']['right_hand_pose']).float() # not fix yet
        # body_pose = vposer.decode(torch.from_numpy(result['front'][0]['result']['body_pose']).float(), output_type='aa').view(1, -1) # decode the body pose
        # jaw_pose, leye_pose, reye_pose, expression = smplx_face_params
        # vt, _ , fv, ft, _ = smplx_uv_params
        
        # # SMPLX model
        # model_smplx =  model_head.smplx_male.model if gender == "male" else model_head.smplx_female.model if gender == "female" else model_head.smplx_neutral.model 
        # # # Get the vertex
        # output = model_smplx(betas=betas,
        #                     # left_hand_pose = left_hand_pose,
        #                     # right_hand_pose = right_hand_pose,
        #                     body_pose = body_pose,
        #                     jaw_pose = jaw_pose,
        #                     leye_pose = leye_pose,
        #                     reye_pose = reye_pose,
        #                     expression = expression,
        #                     return_verts=True)
        # vertices = output.vertices.detach().cpu().numpy().squeeze()
        
        # # Export the obj file
        # with open('modules/outputs/output.obj', 'w') as f:
        #     f.write('mtllib output.mtl\nusemtl output\n')
        #     for x,y,z in vertices:
        #         f.write('v {} {} {}\n'.format(x,y,z))
        #     for u, v in vt:
        #         v = 1.0 - v
        #         f.write('vt {} {}\n'.format(u, v))
        #     for face_vertex, face_texture in zip (fv, ft):
        #         v1, v2, v3 = face_vertex + 1
        #         vt1, vt2, vt3 = face_texture + 1
        #         f.write('f {}/{} {}/{} {}/{}\n'.format(v1, vt1, v2, vt2, v3, vt3))
        # # Write the uvmap to file
        # cv2.imwrite("modules/outputs/output_uvmap.png", uvmap_face)
        
        return HttpResponse("Successfully created the avatar!")
        
        # Combine the complete uvmap with uvmap_face
        # base_64_img_ret = encode_byte_img_base64(uvmap_face)
        # if base_64_img_ret is None:
        #     return HttpResponse("Can not encode uvmap!")
        # a = np.np.array([1,2,3])
        # a.tolist()
        # data = {"uvmap": base_64_img_ret, "jaw_pose": jaw_pose.tolist(), "leye_pose": leye_pose.tolist(), 
        #                 "reye_pose": reye_pose.tolist(), "expression": expression.tolist()}
        # return HttpResponse(json.dumps(data))