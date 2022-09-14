# from smplifyx.utils import JointMapper, smpl_to_openpose
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# /Users/jaydentran1909/Documents/smplx/avatar3d/modules/smplx/model3d.py
from smplx_api.model3d import BaseModel


import cv2
import smplx
from pathlib import Path
import yaml
from smplx_api.smplx_pose.smplifyx.camera import create_camera
from smplx_api.smplx_pose.smplifyx.prior import create_prior
import numpy as np
import torch

try:
    from smplx_api.smplx_pose.openpose.python.openpose import pyopenpose as op
except ImportError as e:
    # print('Error: OpenPose library could not be found. \
    #        Did you enable `BUILD_PYTHON` in CMake and have \
    #        this Python script in the right folder?')
    raise e

cfg_files = {
    'smplx': Path('smplifyx/cfg_files/fit_smplx.yaml'),
    'smplh': Path('smplifyx/cfg_files/fit_smplh.yaml'),
    'smpl': Path('smplifyx/cfg_files/fit_smpl.yaml')
}

class SMPLX_Pose(BaseModel):
    """SMPL-X head model"""
    def __init__(self,
                gender='male', # Choose SMPL gender ['male', 'female', 'neutral']
                model_type='smplx', # Choose SMPL type ['smplx', 'smplh', 'smpl']
                name='SMPLX_Pose'):
        super(SMPLX_Pose, self).__init__(name)

        assert gender in ['male', 'female', 'neutral'], \
               Exception('gender is undefined, make sure it is "male", "female" or "neutral"')
        assert model_type in ['smplx', 'smplh', 'smpl'], \
               Exception('model_type is undefined, make sure it is "smplx", "smplh" or "smpl"')

        with open(str(cfg_files[model_type]), 'r') as f:
            raw_cfg_content = f.read()
        params_cfg = yaml.load(raw_cfg_content, Loader=yaml.SafeLoader)
        params_cfg.gender = gender
        
        ###=== OpenPose model - Extracting keypoints ===###
        self.op_model = op.WrapperPython()
        op_mdls_folder_path = str(Path('openpose/models'))

        # Check model folder exists
        assert os.path.exists(op_mdls_folder_path), \
               Exception(f'Folder {op_mdls_folder_path} not found!')
        assert os.path.exists(os.path.join(op_mdls_folder_path, 'pose')), \
               Exception('Folder named "pose" does not exist. Please place the model file inside folder named "pose"')
        assert (params_cfg.use_hands and os.path.exists(os.path.join(op_mdls_folder_path, 'face'))) or not op_face_enabled, \
               Exception('Folder named "face" does not exist. Please place the model file inside folder named "face"')
        assert (params_cfg.use_face and os.path.exists(os.path.join(op_mdls_folder_path, 'hand'))) or not op_hand_enabled, \
               Exception('Folder named "hand" does not exist. Please place the model file inside folder named "hand"')

        params = {
            'model_folder': op_mdls_folder_path,
            'face': params_cfg.use_hands,
            'hand': params_cfg.use_face
        }
        self.op_model.configure(params)
        self.op_model.start()

        ###=== SMPLify-X model - 2D to 3D mesh ===###
    #     model_params = dict(model_path='smplifyx/models',
    #                     joint_mapper=JointMapper(smpl_to_openpose(
    #                         model_type=model_type,
    #                         use_hands=params_cfg.use_hands,
    #                         use_face=params_cfg.use_face
    #                     )),
    #                     create_global_orient=True,
    #                     create_body_pose=not params_cfg.use_vposer,
    #                     create_betas=True,
    #                     create_left_hand_pose=True,
    #                     create_right_hand_pose=True,
    #                     create_expression=True,
    #                     create_jaw_pose=True,
    #                     create_leye_pose=True,
    #                     create_reye_pose=True,
    #                     create_transl=False,
    #                     **params_cfg)

    #     self.smplx_model = smplx.create(gender=gender, **model_params)
    #     focal_length = params_cfg.focal_length
    #     float_dtype = params_cfg.get('float_dtype', 'float32')
    #     if float_dtype == 'float64':
    #         dtype = torch.float64
    #     elif float_dtype == 'float32':
    #         dtype = torch.float32
    #     else:
    #         raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))
    #     camera = create_camera(focal_length_x=focal_length,
    #                           focal_length_y=focal_length,
    #                           dtype=dtype,
    #                           **params_cfg)
    #     if hasattr(camera, 'rotation'):
    #         camera.rotation.requires_grad = False

    #     use_hands = params_cfg.get('use_hands', True)
    #     use_face = params_cfg.get('use_face', True)

    #     body_pose_prior = create_prior(
    #         prior_type=params_cfg.get('body_prior_type'),
    #         dtype=dtype,
    #         **params_cfg)

    #     jaw_prior, expr_prior = None, None
    #     if use_face:
    #         jaw_prior = create_prior(
    #             prior_type=params_cfg.get('jaw_prior_type'),
    #             dtype=dtype,
    #             **params_cfg)
    #         expr_prior = create_prior(
    #             prior_type=params_cfg.get('expr_prior_type', 'l2'),
    #             dtype=dtype, **params_cfg)

    #     left_hand_prior, right_hand_prior = None, None
    #     if use_hands:
    #         lhand_args = params_cfg.copy()
    #         lhand_args['num_gaussians'] = params_cfg.get('num_pca_comps')
    #         left_hand_prior = create_prior(
    #             prior_type=params_cfg.get('left_hand_prior_type'),
    #             dtype=dtype,
    #             use_left_hand=True,
    #             **lhand_args)

    #         rhand_args = params_cfg.copy()
    #         rhand_args['num_gaussians'] = params_cfg.get('num_pca_comps')
    #         right_hand_prior = create_prior(
    #             prior_type=params_cfg.get('right_hand_prior_type'),
    #             dtype=dtype,
    #             use_right_hand=True,
    #             **rhand_args)

    #     shape_prior = create_prior(
    #         prior_type=params_cfg.get('shape_prior_type', 'l2'),
    #         dtype=dtype, **params_cfg)
        
    #     if params_cfg.use_cuda and torch.cuda.is_available():
    #         device = torch.device('cuda')
    #         camera = camera.to(device=device)
    #         female_model = female_model.to(device=device)
    #         male_model = male_model.to(device=device)
    #         if model_type != 'smplh':
    #             neutral_model = neutral_model.to(device=device)
    #         body_pose_prior = body_pose_prior.to(device=device)
    #         angle_prior = angle_prior.to(device=device)
    #         shape_prior = shape_prior.to(device=device)
    #         if use_face:
    #             expr_prior = expr_prior.to(device=device)
    #             jaw_prior = jaw_prior.to(device=device)
    #         if use_hands:
    #             left_hand_prior = left_hand_prior.to(device=device)
    #             right_hand_prior = right_hand_prior.to(device=device)
    #     else:
    #         device = torch.device('cpu')
    #     angle_prior = create_prior(prior_type='angle', dtype=dtype)
    #     joint_weights = self._get_joint_weights(params_cfg, dtype).to(device=device,
    #                                                    dtype=dtype)
    #     joint_weights.unsqueeze_(dim=0)


    # def _get_joint_weights(self, params_cfg, dtype):
    #     NUM_BODY_JOINTS = 25
    #     NUM_HAND_JOINTS = 20
    #     num_joints = (NUM_BODY_JOINTS +
    #                        2 * NUM_HAND_JOINTS * params_cfg.use_hands)
    #     use_face_contour = False
    #     optim_weights = np.ones(num_joints + 2 * params_cfg.use_hands +
    #                             params_cfg.use_face * 51 +
    #                             17 * use_face_contour,
    #                             dtype=np.float32)

    #     if params_cfg.joints_to_ign is not None and -1 not in params_cfg.joints_to_ign:
    #         optim_weights[params_cfg.joints_to_ign] = 0.
    #     return torch.tensor(optim_weights, dtype=dtype)

    
    # define you custom functions here
    def _preprocess(self, inputs, **kwargs):
        """Preprocesses the inputs.

        Args:
            inputs (tensor/list/tuple): specifies the input to the model.
            **kwargs: additional keyword arguments.
        """
        pass
    
    def _extract_keypoints(self, image_path):
        datum = op.Datum()
        datum.cvInputData = cv2.imread(image_path)
        self.op_model.emplaceAndPop(op.VectorDatum([datum]))
        return {'pose': datum.poseKeypoints,
                'face': datum.faceKeypoints,
                'left_hand': datum.handKeypoints[0],
                'right_hand': datum.handKeypoints[1]}
        

    # the inherited predict function is used to call your custom functions
    def predict(self, inputs, **kwargs):
        """Predicts the output of the model given the inputs.

        Args:
            inputs (tensor/list/tuple): specifies the input to the model.
            **kwargs: additional keyword arguments.
        """
        for path in inputs:
            keypoints = self._extract_keypoints(self, path)
            print(keypoints)

# if __name__ == '__main__':
#     inputs = ['/Users/jaydentran1909/Documents/smplx/avatar3d/modules/smplx/smplx_pose/data/images/007.jpg']
#     model = SMPLX_Pose()
#     print(model.predict(inputs))
