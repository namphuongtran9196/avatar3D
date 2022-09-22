import os
import cv2
import torch
import json
import numpy as np
import shutil
from datetime import datetime
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from human_body_prior.tools.model_loader import load_vposer

from modules.core import decode_byte_img_base64, encode_byte_img_base64
from modules.smplx.smplx_head.model import SMPLX_Head
from modules.smplx.smplx_pose.model import SMPLX_Pose
from modules.smplx.smplx_texture.model import SMPLX_Texture


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

        # Create new input folder, # remove this method in the future
        inputs_folder_name = "inputs" + datetime.now().strftime("%d%m%Y%H%M%S")
        outputs_folder_name = "outputs" + datetime.now().strftime("%d%m%Y%H%M%S")
        
        inputs_folder_path = "modules/smplx/smplx_pose/data/images/" + inputs_folder_name
        output_folder_path = os.path.abspath("modules/outputs/" + outputs_folder_name)
        smplifyx_out_path = os.path.abspath("modules/smplx/smplx_pose/data/smplifyx_" + outputs_folder_name)
        os.makedirs(inputs_folder_path, exist_ok=True)
        os.makedirs(output_folder_path, exist_ok=True)
        os.makedirs(smplifyx_out_path, exist_ok=True)
        
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
        cv2.imwrite(f"{inputs_folder_path}/front.jpg", img_front)
        cv2.imwrite(f"{inputs_folder_path}/back.jpg", img_back)

        # Predict the smplx parameters
        result = model_pose.predict(inputs_folder_name, gender,smplifyx_out_path)
        
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
        with open(f'modules/outputs/{outputs_folder_name}/output.obj', 'w') as f:
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
        
        # Build the complete uvmap

        front_img_path = os.path.abspath(f"{inputs_folder_path}/front.jpg")
        back_img_path = os.path.abspath(f"{inputs_folder_path}/back.jpg")
        uvmap_complete = model_texture.predict(smplifyx_out_path, front_img_path, back_img_path, 'outputs')
        
        # Merge the uvmap_head with the uvmap_complete
        uvmap_complete =  np.where(uvmap_face==0, uvmap_complete, uvmap_face)
        cv2.imwrite(f"modules/outputs/{outputs_folder_name}/output_uvmap.png", uvmap_complete)
        
        # release data
        shutil.rmtree(os.path.abspath("modules/smplx/smplx_pose/data"))
        shutil.rmtree(smplifyx_out_path)
        shutil.copy("modules/outputs/output.mtl", f"modules/outputs/{outputs_folder_name}/output.mtl")
        
        return HttpResponse("Successfully created the avatar!")
        