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
model_head = SMPLX_Head(
                        model_3dffa_config_path=os.path.abspath("modules/smplx/smplx_head/src/V2_3DDFA/configs/mb1_120x120.yml"), 
                        model_smplx_path=os.path.abspath("modules/smplx/smplx_pose/smplifyx/models/smplx"),
                        max_iter=10000,
                        uvmap_path = os.path.abspath("modules/smplx/smplx_head/src/uvmap.obj"),
                        num_expression_coeffs=10,
                        )

model_pose = SMPLX_Pose()
model_texture = SMPLX_Texture()
model_uvmap = SMPLX_UVmap()

# For decode the body pose
vposer, _ = load_vposer(os.path.abspath("modules/smplx/smplx_pose/smplifyx/models/vposer_v1_0"), vp_model='snapshot')
vposer = vposer.to(device='cpu')
vposer.eval()

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
        # result = model_pose.predict("inputs", gender)
    

        result = {'front': [{'loss': 3156.739501953125, 'result': {'camera_rotation': np.array([[[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]]), 'camera_translation': np.array([[-0.00718224,  0.5988984 ,  2.671546  ]]), 'betas': np.array([[ 2.7905169 , -0.68636966,  0.5671541 , -0.17487629,  0.81810755,
            -0.33871022,  0.54144704,  0.04676029, -0.3803052 , -1.1641427 ]],
            ), 'global_orient': np.array([[3.3310313e+00, 2.0705746e-03, 1.0424853e-01]]), 'left_hand_pose': np.array([[ 1.0625746 , -0.522857  ,  0.33080366,  0.6896105 ,  0.17471135,
                        -0.9823367 ,  0.6068878 ,  0.080503  ,  0.46507204, -1.070033  ,
                        0.12440845, -1.712447  ]]), 'right_hand_pose': np.array([[ 1.014201  , -0.31643376,  0.42141536,  1.070554  , -0.18962747,
                        -0.2953818 ,  0.3067134 ,  0.03846581, -0.49314907, -1.3246461 ,
                        1.146475  , -0.93388706]]), 'jaw_pose': np.array([[-0.01214127, -0.0010776 , -0.00040106]]), 'leye_pose': np.array([[-2.061035 , -2.8716369,  0.7404814]]), 'reye_pose': np.array([[-3.0574198,  1.5545638, -0.682958 ]]), 'expression': np.array([[-0.49277523,  0.7018746 , -0.74008894,  0.1733832 , -0.02774926,
                        -0.14176257,  0.20646305,  0.2374246 , -0.2007268 , -0.18161967]],
            ), 'body_pose': np.array([[ 0.31395987,  0.7123356 , -0.25079763,  0.32289875, -0.19875377,
                        0.0804749 ,  3.7290208 , -0.84047556, -0.8148157 ,  0.36138567,
                        1.8127161 , -0.17155695,  2.0222218 ,  0.14143382,  1.7173216 ,
                        -0.8055132 ,  0.07942871, -1.8421651 , -0.0457604 , -0.7970532 ,
                        1.7629416 , -0.7555018 , -1.9692101 , -0.3202928 , -0.9084285 ,
                        -0.5270127 , -0.785477  , -0.23143667, -0.6441702 ,  0.43172768,
                        -1.2823657 , -0.659704  ]])}}]}
        
        # Predict the smplx head based on the smplx parameters
        model_head.set_face_smplify(result['front'][0]['result']['jaw_pose'], 
                                    result['front'][0]['result']['leye_pose'], 
                                    result['front'][0]['result']['reye_pose'], 
                                    result['front'][0]['result']['expression'])
        
        uvmap_face, smplx_face_params, smplx_uv_params = model_head.predict(img_front, gender=gender)
        
        
        # SMPLX parameters
        betas = torch.from_numpy(result['front'][0]['result']['betas']).float()
        left_hand_pose = torch.from_numpy(result['front'][0]['result']['left_hand_pose']).float() # not fix yet
        right_hand_pose = torch.from_numpy(result['front'][0]['result']['right_hand_pose']).float() # not fix yet
        body_pose = vposer.decode(torch.from_numpy(result['front'][0]['result']['body_pose']).float(), output_type='aa').view(1, -1) # decode the body pose
        jaw_pose, leye_pose, reye_pose, expression = smplx_face_params
        vt, _ , fv, ft, _ = smplx_uv_params
        
        # SMPLX model
        model_smplx =  model_head.smplx_male.model if gender == "male" else model_head.smplx_female.model if gender == "female" else model_head.smplx_neutral.model 
        # # Get the vertex
        output = model_smplx(betas=betas,
                            # left_hand_pose = left_hand_pose,
                            # right_hand_pose = right_hand_pose,
                            body_pose = body_pose,
                            jaw_pose = jaw_pose,
                            leye_pose = leye_pose,
                            reye_pose = reye_pose,
                            expression = expression,
                            return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        
        # Export the obj file
        with open('modules/outputs/output.obj', 'w') as f:
            f.write('mtllib output.mtl\nusemtl output\n')
            for x,y,z in vertices:
                f.write('v {} {} {}\n'.format(x,y,z))
            for u, v in vt:
                v = 1.0 - v
                f.write('vt {} {}\n'.format(u, v))
            for face_vertex, face_texture in zip (fv, ft):
                v1, v2, v3 = face_vertex + 1
                vt1, vt2, vt3 = face_texture + 1
                f.write('f {}/{} {}/{} {}/{}\n'.format(v1, vt1, v2, vt2, v3, vt3))
        # Write the uvmap to file
        cv2.imwrite("modules/outputs/output_uvmap.png", uvmap_face)
        
        return HttpResponse("Successfully created the avatar!")
        
        # Combine the complete uvmap with uvmap_face
        base_64_img_ret = encode_byte_img_base64(uvmap_face)
        if base_64_img_ret is None:
            return HttpResponse("Can not encode uvmap!")
        a = np.np.array([1,2,3])
        a.tolist()
        data = {"uvmap": base_64_img_ret, "jaw_pose": jaw_pose.tolist(), "leye_pose": leye_pose.tolist(), 
                        "reye_pose": reye_pose.tolist(), "expression": expression.tolist()}
        return HttpResponse(json.dumps(data))
