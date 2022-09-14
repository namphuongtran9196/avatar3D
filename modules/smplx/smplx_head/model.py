import os
import cv2
import torch
import tqdm
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from modules.smplx.model3d import BaseModel
from torch.autograd import Variable
from .src.utils import *
from .src.V2_3DDFA.V2_3DDFA import Face3DDFA
from .src.models import SMPLX

class SMPLX_Head(BaseModel):
    """SMPL-X head model"""
    def __init__(self, name='SMPLX_Head', 
                        model_3dffa_config_path="./src/V2_3DDFA/configs/mb1_120x120.yml", 
                        model_smplx_path="./models/smplx",
                        max_iter=10000,
                        uvmap_path = "./src/uvmap.obj",
                        device = 'cpu'):
        super(SMPLX_Head, self).__init__(name)
        # Create your model here
        self.model_3dffa = Face3DDFA(model_3dffa_config_path)

        self.smplx_neutral = SMPLX(model_path = model_smplx_path, 
                                    gender='neutral', 
                                    use_face_contour=True,
                                    num_betas=100,
                                    num_expression_coeffs=10,
                                    num_pca_comps = 6,
                                    create_expression = True,
                                    create_jaw_pose = True,
                                    create_leye_pose = True,
                                    create_reye_pose=True,
                                    )
        self.smplx_male = SMPLX(model_path = model_smplx_path, 
                                    gender='male', 
                                    use_face_contour=True,
                                    num_betas=100,
                                    num_expression_coeffs=10,
                                    num_pca_comps = 6,
                                    create_expression = True,
                                    create_jaw_pose = True,
                                    create_leye_pose = True,
                                    create_reye_pose=True,
                                    )
        self.smplx_female = SMPLX(model_path = model_smplx_path, 
                                    gender='female', 
                                    use_face_contour=True,
                                    num_betas=100,
                                    num_expression_coeffs=10,
                                    num_pca_comps = 6,
                                    create_expression = True,
                                    create_jaw_pose = True,
                                    create_leye_pose = True,
                                    create_reye_pose=True,
                                    )
    
        # Remove gradient
        for layer in self.smplx_neutral.parameters():
            layer.requires_grad = False
        for layer in self.smplx_male.parameters():
            layer.requires_grad = False
        for layer in self.smplx_female.parameters():
            layer.requires_grad = False
        
        
        self.device = device
        self.to_device(torch.device(self.device))
        self.max_iter = max_iter
        self.uvmap_path = uvmap_path
        
    def to_device(self,device):
        self.smplx_neutral.to(device)
        self.smplx_male.to(device)
        self.smplx_female.to(device)

    def _init_variable(self):
        
        # Set dtype to float32
        dtype = torch.cuda.FloatTensor if self.device == 'gpu' else torch.FloatTensor
        
        # Create dummy input data without gradients
        dummy_input = Variable(torch.ones(1, 1).type(dtype), requires_grad=False)

        # Control the model shape, like thin or fat. [Batch_size , num_betas]  
        betas = torch.zeros([1, self.smplx_neutral.num_betas]).type(dtype)
        # Control the hand pose. [Batch_size, num_pca_comps]
        left_hand_pose = torch.zeros([1, self.smplx_neutral.num_pca_comps]).type(dtype)
        right_hand_pose = torch.zeros([1, self.smplx_neutral.num_pca_comps]).type(dtype)
        # Control the model pose. [Batch_size, 3]
        body_pose = torch.zeros([1, self.smplx_neutral.NUM_BODY_JOINTS * 3]).type(dtype)

        # Create torch variables
        global_orient_variable = Variable(torch.zeros([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
        jaw_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
        leye_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True)  # [Batch_size, 3] [x, y]
        reye_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
        transl_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
        expression_variable = Variable(torch.randn([1, self.smplx_neutral.num_expression_coeffs]).type(dtype), requires_grad=True) # [Batch_size, num_expression_coeffs]
        
        return dummy_input,betas, left_hand_pose, right_hand_pose, global_orient_variable, jaw_pose_variable, \
                                    leye_pose_variable, reye_pose_variable, body_pose, transl_variable, expression_variable

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
        lmk3D_true = torch.from_numpy(lmk3D_true).float().to(torch.device(self.device))

        return lmk3D_true, img_crop
    
    def _predict_smplx(self, model, dummy_input,betas, left_hand_pose, right_hand_pose, global_orient_variable, jaw_pose_variable, \
                                                leye_pose_variable, reye_pose_variable, body_pose, transl_variable, expression_variable):
        """
        Predict SMPL-X model from variables
        """
        # Dot product
        global_orient_pred = dummy_input.mm(global_orient_variable)
        jaw_pose_pred = dummy_input.mm(jaw_pose_variable)
        leye_pose_pred = dummy_input.mm(leye_pose_variable)
        reye_pose_pred = dummy_input.mm(reye_pose_variable)
        expression_pred = dummy_input.mm(expression_variable)
        transl_pred = dummy_input.mm(transl_variable)

        output = model(
                betas=betas,
                body_pose=body_pose,
                left_hand_pose = left_hand_pose,
                right_hand_pose = right_hand_pose,
                global_orient=global_orient_pred,
                expression=expression_pred,
                jaw_pose = jaw_pose_pred,
                leye_pose = leye_pose_pred,
                reye_pose = reye_pose_pred,
                transl = transl_pred,
                    return_verts=True,)

        return output
    # the inherited predict function is used to call your custom functions
    def predict(self, inputs, gender='neutral', **kwargs):
        """Predicts the output of the model given the inputs.

        Args:
            inputs (tensor/list/tuple): specifies the input to the model.
            **kwargs: additional keyword arguments.
        """
        assert gender in ['neutral', 'male', 'female'], "Gender must be neutral, male or female"
        # Set dtype to float32
        device = torch.device(self.device)
        dtype = torch.cuda.FloatTensor if self.device == 'gpu' else torch.FloatTensor

        # Get model
        model = self.smplx_neutral if gender == 'neutral' else self.smplx_male if gender == 'male' else self.smplx_female
        
        # Init variable
        dummy_input,betas, left_hand_pose, right_hand_pose, global_orient_variable, jaw_pose_variable, \
            leye_pose_variable, reye_pose_variable, body_pose, transl_variable, expression_variable = self._init_variable()
        
        # Clone the image
        img_raw = inputs.copy()
        # Predict the 3D face landmarks
        logging.info("Predicting 3D face landmarks...")
        lmk3D_true, img_crop = self._predict_3dffa(img_raw)
        # Copy the cropped image for create the UV map
        source_img = img_crop.copy()
        
        # Create first pass
        output = self._predict_smplx(model, dummy_input,betas, left_hand_pose, right_hand_pose, global_orient_variable, jaw_pose_variable, \
                                                    leye_pose_variable, reye_pose_variable, body_pose, transl_variable, expression_variable)
        
        # Calculate camera parameters
        s2d = torch.mean(torch.norm(lmk3D_true[:,:2] -torch.mean(lmk3D_true[:,:2], dim=0), dim=1))
        s3d = torch.mean(torch.sqrt(torch.sum(torch.square(output.joints.squeeze()[76:,:]-torch.mean(output.joints.squeeze()[76:,:], axis=0))[:, :2], dim=1)))    
        torch_scale_variable = Variable((s2d/s3d).type(dtype), requires_grad=True)


        # Optimize the parameters of the model
        learning_rate = 0.001
        best_loss = 999999
        loss = 999999
        step = 0
        global_orient, jaw_pose, leye_pose, reye_pose, transl, expression, torch_scale = None, None, None, None, None, None, None

        logging.info("Optimizing the parameters of the smplx model...")
        while loss > 0.001:
            step+=1
            # Forward pass
            output = self._predict_smplx(model, dummy_input,betas, left_hand_pose, right_hand_pose, global_orient_variable, jaw_pose_variable, \
                                                    leye_pose_variable, reye_pose_variable, body_pose, transl_variable, expression_variable)

            # Get the 3D face landmarks of smplx
            lmk3D_pred = output.joints.squeeze()[76:,:]
            # Project the 3D face landmarks to 2D
            lmks_proj_2d = torch_project_points(lmk3D_pred, torch_scale_variable, torch.zeros(2).to(device), device)
            # Calculate the MSE loss between the 2D landmarks
            loss = (lmks_proj_2d - lmk3D_true[:,:2]).pow(2).sum()
            
            if step % 500 == 0:
                # Save the best parameters
                if loss < best_loss:
                    best_loss = loss
                    global_orient = dummy_input.mm(global_orient_variable).data
                    jaw_pose = dummy_input.mm(jaw_pose_variable).data.detach()
                    leye_pose = dummy_input.mm(leye_pose_variable).data.detach()
                    reye_pose = dummy_input.mm(reye_pose_variable).data.detach()
                    transl = dummy_input.mm(transl_variable).data.detach()
                    expression = dummy_input.mm(expression_variable).data.detach()
                    torch_scale = torch_scale_variable.data.detach()
                    logging.info("{} Current loss: {}".format(step, loss.detach().cpu().numpy()))
            if np.isnan(loss.detach().cpu().numpy()):
                logging.info("Reinitializing because of NaN")
                dummy_input,betas, left_hand_pose, right_hand_pose, global_orient_variable, jaw_pose_variable, \
                                    leye_pose_variable, reye_pose_variable, transl_variable, expression_variable = self._init_variable()
                loss = 99999999
                continue

            # Use autograd to compute the backward pass.
            loss.backward()

            # Update weights using gradient descent;
            global_orient_variable.data -= learning_rate * global_orient_variable.grad.data
            jaw_pose_variable.data -= learning_rate * jaw_pose_variable.grad.data
            leye_pose_variable.data -= learning_rate * leye_pose_variable.grad.data
            reye_pose_variable.data -= learning_rate * reye_pose_variable.grad.data
            expression_variable.data -= learning_rate * expression_variable.grad.data
            transl_variable.data -= learning_rate * transl_variable.grad.data
            torch_scale_variable.data -= learning_rate * torch_scale_variable.grad.data

            # Manually zero the gradients after updating weights
            global_orient_variable.grad.data.zero_()
            jaw_pose_variable.grad.data.zero_()
            leye_pose_variable.grad.data.zero_()
            reye_pose_variable.grad.data.zero_()
            expression_variable.grad.data.zero_()
            transl_variable.grad.data.zero_()
            torch_scale_variable.grad.data.zero_()
            
            if step > self.max_iter:
                break
        
        logging.info("Creating UV map")
        # Forward pass
        output = output = model(
                betas=betas,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                body_pose=body_pose,
                global_orient=global_orient,
                expression=expression,
                jaw_pose = jaw_pose,
                leye_pose = leye_pose,
                reye_pose = reye_pose,
                transl = transl,
                    return_verts=True,)
        
        # Project the 3D face landmarks to 2D
        proj_2d_points = torch_project_points(output.vertices[0], torch_scale, torch.zeros(2), torch.device('cpu'))

        # Change the y axis
        proj_2d_points[:, 1] = 1 - proj_2d_points[:, 1]
        proj_2d_points = proj_2d_points.detach().numpy().astype(np.float32)

        vt,vn , fv, ft, fn  = load_uvmap(self.uvmap_path)
        
        # create the UV map
        uvmap_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
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
                
                
                valid_point = True
                for x, y in [xy1, xy2, xy3]:
                    if x < 0 or x >= source_img.shape[1] or y < 0 or y >= source_img.shape[0]:
                        valid_point = False
                if not valid_point:
                    continue

                tri1 = np.float32([[xy1, xy2, xy3]])
                tri2 = np.float32([[uv1, uv2, uv3]])

                black_img = np.zeros((1000, 1000, 3), dtype=np.uint8)

                single_trimesh_texture = transfer_single_trimesh_texture(source_img, black_img, tri1, tri2)
            
                uvmap_img = np.where(single_trimesh_texture != 0, single_trimesh_texture, uvmap_img)
        
        return uvmap_img, (jaw_pose, leye_pose, reye_pose, expression)