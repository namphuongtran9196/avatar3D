import os
import cv2
import torch
import tqdm
import numpy as np
import logging
import torch.nn as nn
logging.basicConfig(level=logging.INFO)

from modules.smplx.model3d import BaseModel
from torch.autograd import Variable
from .src.utils import *
from .src.V2_3DDFA.V2_3DDFA import Face3DDFA
from .src.models import SMPLX

class SMPLX_Fitting(nn.Module):
    def __init__(self, smplx_path="models/smplx",gender='neutral',use_face_contour=True,num_expression_coeffs=10, device='cpu'):
        super(SMPLX_Fitting, self).__init__()
        # The model gender
        gender = gender # 'neutral' or 'male' or 'female'
        # Whether to compute the keypoints that form the facial contour
        use_face_contour=use_face_contour
                    
        self.model = SMPLX(
                            model_path = smplx_path, 
                            gender=gender, 
                            use_face_contour=use_face_contour,
                            num_expression_coeffs=num_expression_coeffs,
                            )
        for layer in self.model.parameters():
            layer.requires_grad = False
            
        self.dtype = torch.cuda.FloatTensor if device=='gpu' else torch.FloatTensor
        self._init_variables()
        self.use_face_from_smplify = False
        
    def _init_variables(self):
        self.global_orient_variable = torch.nn.Parameter(torch.ones((1,3)).type(self.dtype) * 0.0001, requires_grad=True)
        # self.betas_variable = torch.nn.Parameter(torch.ones((1, self.model.num_betas)).type(self.dtype) * 0.0001, requires_grad=True)
        self.jaw_pose_variable = torch.nn.Parameter(torch.ones((1,3)).type(self.dtype) * 0.0001, requires_grad=True)
        self.leye_pose_variable = torch.nn.Parameter(torch.ones((1, 3)).type(self.dtype) * 0.0001, requires_grad=True)
        self.reye_pose_variable = torch.nn.Parameter(torch.ones((1, 3)).type(self.dtype) * 0.0001, requires_grad=True)
        self.transl_variable = torch.nn.Parameter(torch.ones((1, 3)).type(self.dtype)* 0.0001, requires_grad=True)
        self.expression_variable = torch.nn.Parameter(torch.ones((1, self.model.num_expression_coeffs)).type(self.dtype)* 0.0001, requires_grad=True)
        self.scale_axis_z = torch.nn.Parameter(torch.ones((1, 1)).type(self.dtype)* 1000, requires_grad=True)
        # self.body_pose_variable = torch.nn.Parameter(torch.ones((1, self.model.NUM_BODY_JOINTS * 3)).type(self.dtype)* 0.0001, requires_grad=True)
    
    def set_target_2d_lmks(self, target_2d_lmks):
        self.target_2d_lmks = target_2d_lmks
        
    def set_face_smplify(self, jaw_pose, leye_pose, reye_pose, expression):
        self.use_face_from_smplify = True
        self.jaw_pose = torch.tensor(jaw_pose).type(self.dtype)
        self.leye_pose = torch.tensor(leye_pose).type(self.dtype)
        self.reye_pose = torch.tensor(reye_pose).type(self.dtype)
        self.expression = torch.tensor(expression).type(self.dtype)
        
        
    def forward(self, x):
        global_orient = x.mm(self.global_orient_variable)
        tranlsl = x.mm(self.transl_variable)
        leye_pose = x.mm(self.leye_pose_variable)
        reye_pose = x.mm(self.reye_pose_variable)
        expression = x.mm(self.expression_variable)
        # betas = x.mm(self.betas_variable)
        if self.use_face_from_smplify:
            jaw_pose = self.jaw_pose
            # leye_pose = self.leye_pose
            # reye_pose = self.reye_pose
            # expression = self.expression
        else:
            jaw_pose = x.mm(self.jaw_pose_variable)
        # body_pose = x.mm(self.body_pose_variable)
        scale_z = x.mm(self.scale_axis_z)
        
        output = self.model(global_orient=global_orient,
                            # betas=betas,
                            jaw_pose=jaw_pose,
                            leye_pose=leye_pose,
                            reye_pose=reye_pose,
                            transl=tranlsl,
                            expression=expression,
                            # body_pose=body_pose,
                            return_verts=True)
        
        # Compute scale variables.
        s2d = torch.mean(torch.norm(self.target_2d_lmks -torch.mean(self.target_2d_lmks, dim=0), dim=1))
        s3d = torch.mean(torch.sqrt(torch.sum(torch.square(output.joints.squeeze()[76:,:]-torch.mean(output.joints.squeeze()[76:,:], axis=0))[:, :2], dim=1)))
        
        return output, s2d/s3d, scale_z, (jaw_pose, leye_pose, reye_pose, expression)

class SMPLX_Head(BaseModel):
    """SMPL-X head model"""
    def __init__(self, name='SMPLX_Head', 
                        model_3dffa_config_path="./src/V2_3DDFA/configs/mb1_120x120.yml", 
                        model_smplx_path="./models/smplx",
                        max_iter=10000,
                        uvmap_path = "./src/uvmap.obj",
                        num_expression_coeffs=10):
        super(SMPLX_Head, self).__init__(name)
        # Create your model here
        self.model_3dffa = Face3DDFA(model_3dffa_config_path)

        
        self.smplx_neutral = SMPLX_Fitting(smplx_path=model_smplx_path,
                                            gender='neutral',
                                            use_face_contour=True,
                                            num_expression_coeffs=num_expression_coeffs,
                                            device='cpu')

        self.smplx_male = SMPLX_Fitting(smplx_path=model_smplx_path,
                                            gender='male',
                                            use_face_contour=True,
                                            num_expression_coeffs=num_expression_coeffs,
                                            device='cpu')
        
        self.smplx_female = SMPLX_Fitting(smplx_path=model_smplx_path,
                                            gender='female',
                                            use_face_contour=True,
                                            num_expression_coeffs=num_expression_coeffs,
                                            device='cpu')
        self.max_iter = max_iter
        self.uvmap_path = uvmap_path
    def set_face_smplify(self, jaw_pose, leye_pose, reye_pose, expression):
        self.smplx_female.set_face_smplify(jaw_pose, leye_pose, reye_pose, expression)
        self.smplx_male.set_face_smplify(jaw_pose, leye_pose, reye_pose, expression)
        self.smplx_neutral.set_face_smplify(jaw_pose, leye_pose, reye_pose, expression)
    
    def _predict_3dffa(self, img):
        """
        Predict 3D face from 2D image
        """
        # Predict landmarks and face bounding box
        bboxes, landmarks_3D = self.model_3dffa.predict(img)
        # Crop the face
        bbox = bboxes[0]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        w_ex, h_ex = w * 0.2, h * 0.2
        img_crop = img[int(bbox[1] - h_ex):int(bbox[3] + h_ex), int(bbox[0] - w_ex):int(bbox[2] + w_ex)]

        # Transform landmarks to the cropped image
        lmk3D = landmarks_3D[0].T
        lmk3D[:, 0] = lmk3D[:, 0] - bbox[0] + w_ex
        lmk3D[:, 1] = lmk3D[:, 1] - bbox[1] + h_ex

        lmk3D_true = np.zeros_like(lmk3D)

        # Change the landmarks order to fit the SMPL-X model
        lmk3D_true[:10, :] = lmk3D[17:27, :] # brow
        lmk3D_true[10:19, :] = lmk3D[27:36,:] # nose
        lmk3D_true[19:31, :] = lmk3D[36:48,:] # eye
        lmk3D_true[31:51, :] = lmk3D[48:68,:]# lip
        lmk3D_true[51:,:] = lmk3D[:17, :] # jaw

        # normalize
        lmk3D_true[:,1] = img_crop.shape[0] - lmk3D_true[:,1] # invert y axis
        lmk3D_true = lmk3D_true / np.array([img_crop.shape[1], img_crop.shape[0], np.max(lmk3D_true[:, 2])])
        lmk3D_true = torch.from_numpy(lmk3D_true).float()

        return lmk3D_true, img_crop
    
    # the inherited predict function is used to call your custom functions
    def predict(self, inputs, gender='neutral', **kwargs):
        """Predicts the output of the model given the inputs.

        Args:
            inputs (tensor/list/tuple): specifies the input to the model.
            **kwargs: additional keyword arguments.
        """
        assert gender in ['neutral', 'male', 'female'], "Gender must be neutral, male or female"
        # Set dtype to float32
        device = torch.device('cpu')
        dtype = torch.FloatTensor
        # Get model
        model = self.smplx_neutral if gender == 'neutral' else self.smplx_male if gender == 'male' else self.smplx_female
        
        # Clone the image
        img_raw = inputs.copy()
        # Predict the 3D face landmarks
        logging.info("Predicting 3D face landmarks...")
        lmk3D_true, img_crop = self._predict_3dffa(img_raw)
        model.set_target_2d_lmks(lmk3D_true[:, :2])
        
        # Copy the cropped image for create the UV map
        source_img = img_crop.copy()
        
        # Create the optimizer and loss function
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        logging.info("Optimizing the parameters of the smplx model...")
        # dummy input for the optimizer
        dummy_input = torch.ones((1, 1)).float()
        for step in range(self.max_iter):
            if step > self.max_iter * 0.5 and step < self.max_iter * 0.8:
                for g in optimizer.param_groups:
                    g['lr'] = 1e-3
            elif step > self.max_iter*0.4:
                for g in optimizer.param_groups:
                    g['lr'] = 1e-4
            # Forward pass: Compute predicted y by passing x to the model
            y_pred, torch_scale, scale_z, _ = model(dummy_input)
            
            y_pred_proj = torch_project_points(y_pred.joints.squeeze()[76:,:], torch_scale, torch.zeros(2), torch.device('cpu'))
            # Compute and print loss
            loss1 = criterion(y_pred_proj.flatten(), lmk3D_true[:,:2].flatten())
            loss2 = criterion(y_pred.joints.squeeze()[76:,2:].flatten(), lmk3D_true[:,2:].flatten() / scale_z)
            loss = loss1 + 0.2 * loss2
                
            if step % 100 == 99:
                logging.info("Step: {} , Loss: {}".format(step, loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        logging.info("Creating UV map")
        # Forward pass
        output, torch_scale,_, (jaw_pose, leye_pose, reye_pose, expression) = model(dummy_input)
        
        # Project the 3D face landmarks to 2D
        proj_2d_points = torch_project_points(output.vertices[0], torch_scale, torch.zeros(2), torch.device('cpu'))
        # Change the y axis
        proj_2d_points[:, 1] = 1 - proj_2d_points[:, 1]
        proj_2d_points = proj_2d_points.detach().numpy().astype(np.float32)
        
        # Load the UV map
        vt,vn , fv, ft, fn  = load_uvmap(self.uvmap_path)
        
        # load source image
        texture_img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        for i in tqdm.tqdm(range(len(fv))):
            # get vertices of the face
            v1, v2, v3 = fv[i]
            # get the corresponding uv index
            index_uv1, index_uv2, index_uv3 = ft[i]
            uv1, uv2, uv3 = vt[index_uv1], vt[index_uv2], vt[index_uv3]

            # get the corresponding normal index
            index_n1, index_n2, index_n3 = fn[i]
            n1, n2, n3 = vn[index_n1], vn[index_n2], vn[index_n3]
            
            if n1[2] > 0 and n2[2] > 0 and n3[2] > 0:
                # convert uv to pixel
                uv1 = [int(uv1[0] * 1000), int(uv1[1] * 1000)]
                uv2 = [int(uv2[0] * 1000), int(uv2[1] * 1000)]
                uv3 = [int(uv3[0] * 1000), int(uv3[1] * 1000)]
                
                # convert xy to pixel
                xy1 = [int(proj_2d_points[v1][0] * source_img.shape[1]), int(proj_2d_points[v1][1] * source_img.shape[0])]
                xy2 = [int(proj_2d_points[v2][0] * source_img.shape[1]), int(proj_2d_points[v2][1] * source_img.shape[0])]
                xy3 = [int(proj_2d_points[v3][0] * source_img.shape[1]), int(proj_2d_points[v3][1] * source_img.shape[0])]
                
                # get valid point based on the vertex normal
                valid_point = True
                for x, y in [xy1, xy2, xy3]:
                    if x < 0 or x >= source_img.shape[1] or y < 0 or y >= source_img.shape[0]:
                        valid_point = False
                if not valid_point:
                    continue
                
                # Copy triangle from source image to texture image
                tri1 = np.float32([[xy1, xy2, xy3]])
                tri2 = np.float32([[uv1, uv2, uv3]])
                black_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
                single_trimesh_texture = transfer_single_trimesh_texture(source_img, black_img, tri1, tri2)
                texture_img = np.where(single_trimesh_texture != 0, single_trimesh_texture, texture_img)
        
        return texture_img, (jaw_pose, leye_pose, reye_pose, expression), (vt, vn , fv, ft, fn)