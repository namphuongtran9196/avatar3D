from modules.smplx.smplx_pose.model import ABS_DIR_PATH
from modules.smplx.smplx_pose.smplifyx.utils import JointMapper, smpl_to_openpose
from modules.smplx.smplx_pose.smplifyx.camera import create_camera
from modules.smplx.smplx_pose.smplifyx.prior import create_prior
from modules.smplx.smplx_pose.smplifyx.fit_single_frame import fit_single_frame
import yaml
import smplx
import torch
import numpy as np
import cv2
import json
import os
import re
from modules.smplx.smplx_pose.smplifyx.cmd_parser import parse_config
from collections import namedtuple

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)

ABS_DIR_PATH = os.path.dirname(__file__)

class SMPLifyXModel:
    def __init__(self, cfg: str):
        with open(cfg, 'r') as f:
            raw_cfg_input = f.read()
        
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        params_cfg = yaml.load(raw_cfg_input, Loader=loader)
        params_cfg = {**params_cfg, **parse_config()}
        
        
        params_cfg['vposer_ckpt'] = os.path.join(ABS_DIR_PATH, 'models', 'vposer_v1_0')
        self.model_params = dict(
            model_path=os.path.join(ABS_DIR_PATH, 'models'),
            joint_mapper=JointMapper(smpl_to_openpose(
                model_type=params_cfg['model_type'],
                use_hands=params_cfg['use_hands'],
                use_face=params_cfg['use_face']
            )),
            create_global_orient=True,
            create_body_pose=not params_cfg['use_vposer'],
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=False,
            **params_cfg
        )

        self.model_params['gender'] = 'male'
        self.male_model = smplx.create(**self.model_params)
        # SMPL-H has no gender-neutral model
        if self.model_params.get('model_type') != 'smplh':
            self.model_params['gender'] = 'neutral'
            self.neutral_model = smplx.create(**self.model_params)
        self.model_params['gender'] = 'female'
        self.female_model = smplx.create(**self.model_params)

        float_dtype = self.model_params.get('float_dtype', 'float32')
        if float_dtype == 'float64':
            self.dtype = torch.float64
        elif float_dtype == 'float32':
            self.dtype = torch.float32
        else:
            raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))
        
        focal_length = self.model_params.get('focal_length')
        self.camera = create_camera(focal_length_x=float(focal_length),
                                focal_length_y=float(focal_length),
                                dtype=self.dtype,
                                **self.model_params)

        if hasattr(self.camera, 'rotation'):
            self.camera.rotation.requires_grad = False

        use_hands = self.model_params.get('use_hands', True)
        use_face = self.model_params.get('use_face', True)

        self.body_pose_prior = create_prior(
            prior_type=self.model_params.get('body_prior_type'),
            dtype=self.dtype,
            **self.model_params)

        self.jaw_prior, self.expr_prior = None, None
        if use_face:
            self.jaw_prior = create_prior(
                prior_type=self.model_params.get('jaw_prior_type'),
                dtype=self.dtype,
                **self.model_params)
            self.expr_prior = create_prior(
                prior_type=self.model_params.get('expr_prior_type', 'l2'),
                dtype=self.dtype, **self.model_params)

        self.left_hand_prior, self.right_hand_prior = None, None
        if use_hands:
            lhand_args = self.model_params.copy()
            lhand_args['num_gaussians'] = self.model_params.get('num_pca_comps')
            self.left_hand_prior = create_prior(
                prior_type=self.model_params.get('left_hand_prior_type'),
                dtype=self.dtype,
                use_left_hand=True,
                **lhand_args)

            rhand_args = self.model_params.copy()
            rhand_args['num_gaussians'] = self.model_params.get('num_pca_comps')
            self.right_hand_prior = create_prior(
                prior_type=self.model_params.get('right_hand_prior_type'),
                dtype=self.dtype,
                use_right_hand=True,
                **rhand_args)

        self.shape_prior = create_prior(
            prior_type=self.model_params.get('shape_prior_type', 'l2'),
            dtype=self.dtype, **self.model_params)

        self.angle_prior = create_prior(prior_type='angle', dtype=self.dtype, **self.model_params)

        # Run on CPU, configure to run on GPU later
        device = torch.device('cpu')

        self.joint_weights = self._get_joint_weights(self.model_params, self.dtype).to(device=device, dtype=self.dtype)
        self.joint_weights.unsqueeze_(dim=0)
    
    def fit(self, image_path, keypoint_path, gender:str='male'):
        img = cv2.imread(image_path).astype(np.float32)[:, :, ::-1] / 255.0
        keyp_tuple = self.read_keypoints(keypoint_path)
        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)

        body_model = None
        if gender == 'male':
            body_model = self.male_model
        elif gender == 'female':
            body_model = self.female_model
        elif gender == 'neutral':
            body_model = self.neutral_model

        results = fit_single_frame(
            img, keypoints,
            body_model=body_model,
            camera=self.camera,
            joint_weights=self.joint_weights,
            dtype=self.dtype,
            # output_folder=output_folder,
            # result_folder=curr_result_folder,
            # out_img_fn=out_img_fn,
            # result_fn=curr_result_fn,
            # mesh_fn=curr_mesh_fn,
            shape_prior=self.shape_prior,
            expr_prior=self.expr_prior,
            body_pose_prior=self.body_pose_prior,
            left_hand_prior=self.left_hand_prior,
            right_hand_prior=self.right_hand_prior,
            jaw_prior=self.jaw_prior,
            angle_prior=self.angle_prior,
            **self.model_params
        )
        return results

    def read_keypoints(self, keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):

        with open(keypoint_fn) as keypoint_file:
            data = json.load(keypoint_file)

        keypoints = []

        gender_pd = []
        gender_gt = []
        for idx, person_data in enumerate(data['people']):
            body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                    dtype=np.float32)
            body_keypoints = body_keypoints.reshape([-1, 3])
            if use_hands:
                left_hand_keyp = np.array(
                    person_data['hand_left_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])
                right_hand_keyp = np.array(
                    person_data['hand_right_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])

                body_keypoints = np.concatenate(
                    [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
            if use_face:
                # TODO: Make parameters, 17 is the offset for the eye brows,
                # etc. 51 is the total number of FLAME compatible landmarks
                face_keypoints = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

                contour_keyps = np.array(
                    [], dtype=body_keypoints.dtype).reshape(0, 3)
                if use_face_contour:
                    contour_keyps = np.array(
                        person_data['face_keypoints_2d'],
                        dtype=np.float32).reshape([-1, 3])[:17, :]

                body_keypoints = np.concatenate(
                    [body_keypoints, face_keypoints, contour_keyps], axis=0)

            if 'gender_pd' in person_data:
                gender_pd.append(person_data['gender_pd'])
            if 'gender_gt' in person_data:
                gender_gt.append(person_data['gender_gt'])

            keypoints.append(body_keypoints)

        return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                        gender_gt=gender_gt)


    def _get_joint_weights(self, params_cfg, dtype):
        NUM_BODY_JOINTS = 25
        NUM_HAND_JOINTS = 20
        num_joints = (NUM_BODY_JOINTS +
                        2 * NUM_HAND_JOINTS * params_cfg['use_hands'])
        use_face_contour = False
        optim_weights = np.ones(num_joints + 2 * params_cfg['use_hands'] +
                            params_cfg['use_face'] * 51 +
                            17 * use_face_contour,
                            dtype=np.float32)

        if params_cfg['joints_to_ign'] is not None and -1 not in params_cfg['joints_to_ign']:
            optim_weights[params_cfg['joints_to_ign']] = 0.
        return torch.tensor(optim_weights, dtype=dtype)
    
    
    def __str__(self):
        return str(self.model_params)

if __name__ == '__main__':
    model = SMPLifyXModel('cfg_files/fit_smplx.yaml')
    print(model)