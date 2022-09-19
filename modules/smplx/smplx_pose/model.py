import os
from modules.smplx.model3d import BaseModel
from modules.smplx.smplx_pose.smplifyx.SMPLifyXModel import SMPLifyXModel as XModel

ABS_DIR_PATH = os.path.dirname(__file__)
BASE_DIR_NAME = {
    'img': os.path.join(ABS_DIR_PATH, 'data/images'),
    'kp': os.path.join(ABS_DIR_PATH, 'data/keypoints'),
    'op_img': os.path.join(ABS_DIR_PATH,'data/openpose_images')
}
SMPL_CONFIG_FILE = {
    'smpl': os.path.join(ABS_DIR_PATH, 'smplifyx/cfg_files/fit_smpl.yaml'),
    'smplh': os.path.join(ABS_DIR_PATH, 'smplifyx/cfg_files/fit_smplh.yaml'),
    'smplx': os.path.join(ABS_DIR_PATH, 'smplifyx/cfg_files/fit_smplx.yaml')
}


class SMPLX_Pose(BaseModel):
    """SMPL-X head model"""
    def __init__(self, # Choose SMPL gender ['male', 'female', 'neutral']
                model_type='smplx', # Choose SMPL type ['smplx', 'smplh', 'smpl']
                name='SMPLX_Pose'):
        super(SMPLX_Pose, self).__init__(name)

        assert model_type in ['smplx', 'smplh', 'smpl'], \
               Exception('model_type is undefined, make sure it is "smplx", "smplh" or "smpl"')
        
        self.model_type = model_type
        self.xmodel = XModel(SMPL_CONFIG_FILE[model_type])
    
    # the inherited predict function is used to call your custom functions
    def predict(self, dir_name, gender, **kwargs):
        """Predicts the output of the model given the inputs.

        Args:
            inputs (tensor/list/tuple): specifies the input to the model.
            **kwargs: additional keyword arguments.
        """

        ### Extract OpenPose keypoints
        image_dir = os.path.join(BASE_DIR_NAME['img'], dir_name)
        keypoint_dir = os.path.join(BASE_DIR_NAME['kp'], dir_name)
        # keypoint_images_dir = os.path.join(BASE_DIR_NAME['kp_img'], dir_name)
        
        OPENPOSE_BIN = os.path.join(ABS_DIR_PATH, 'openpose', 'build', 'examples', 'openpose', 'openpose.bin')
        cmd = f'{OPENPOSE_BIN} --image-dir {image_dir} --write_json {keypoint_dir} --face --hand --display 0 --render-pose 0'
        os.system(cmd)

        ### SMPLifyX - Convert 2D to 3D meshes
        # Next step
        # for x in image_dir:
        #     img_path, keypoint_path
        #     self.xmodel.fit(img_path, keypoint_path)