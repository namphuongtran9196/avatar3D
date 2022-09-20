from genericpath import exists
import os
from modules.smplx.model3d import BaseModel
from modules.smplx.smplx_pose.smplifyx.SMPLifyXModel import SMPLifyXModel as XModel

ABS_DIR_PATH = os.path.dirname(__file__)
BASE_DIR_NAME = {
    'img': os.path.join(ABS_DIR_PATH, 'data/images'),
    'kp': os.path.join(ABS_DIR_PATH, 'data/keypoints'),
    # 'op_img': os.path.join(ABS_DIR_PATH,'data/openpose_images')
}
SMPL_CONFIG_FILE = {
    'smpl': os.path.join(ABS_DIR_PATH, 'smplifyx/cfg_files/fit_smpl.yaml'),
    'smplh': os.path.join(ABS_DIR_PATH, 'smplifyx/cfg_files/fit_smplh.yaml'),
    'smplx': os.path.join(ABS_DIR_PATH, 'smplifyx/cfg_files/fit_smplx.yaml')
}


class SMPLX_Pose(BaseModel):
    """SMPL-X head model"""
    def __init__(self,
                model_type: str = 'smplx',
                name: str = 'SMPLX_Pose'):
        """
        Args:
            model_type (str): Input the model type (smplx, smpl, smplh)
            name (str): name of the base model
        """
        super(SMPLX_Pose, self).__init__(name)

        assert model_type in ['smplx', 'smplh', 'smpl'], \
               Exception('model_type is undefined, make sure it is "smplx", "smplh" or "smpl"')
        
        self.model_type = model_type

        assert os.path.exists(SMPL_CONFIG_FILE[model_type]), Exception('The config file does not found, please make sure it exists!')
        self.xmodel = XModel(SMPL_CONFIG_FILE[model_type])

        for data_folder in BASE_DIR_NAME.values():
            if not os.path.exists(data_folder):
                os.makedirs(data_folder, exist_ok=True)
    
    def predict(self, dir_name: str, gender: str, **kwargs):
        """Predicts the output of the model given the inputs.

        Args:
            dir_name (str): the directory name which includes list of images (inside data/...)
            gender (str): gender of the model ('male', 'female', 'neutral')
            **kwargs: additional keyword arguments.
        """

        ### Extract OpenPose keypoints
        print('Extracting OpenPose keypoints...')

        image_dir = os.path.join(BASE_DIR_NAME['img'], dir_name)
        assert os.path.join(image_dir), Exception(f'{image_dir} does not found!')
        keypoint_dir = os.path.join(BASE_DIR_NAME['kp'], dir_name)
        if not os.path.exists(keypoint_dir):
            os.makedirs(keypoint_dir, exist_ok=True)
        
        OPENPOSE_DIR = os.path.join(ABS_DIR_PATH, 'openpose')
        OPENPOSE_BIN = os.path.join('.', 'build', 'examples', 'openpose', 'openpose.bin')
        cmd = f'cd {OPENPOSE_DIR} && {OPENPOSE_BIN} --image-dir {image_dir} --write_json {keypoint_dir} --face --hand --display 0 --render-pose 0'
        os.system(cmd)

        ### SMPLifyX - Convert 2D to 3D meshes
        print('Converting 2D images to 3D meshes...')
        mesh_results = dict()
        for image, keypoint in zip(sorted(os.listdir(image_dir)), sorted(os.listdir(keypoint_dir))):
            print(f'- Converting {image}...')
            image_abs_path = os.path.join(image_dir, image)
            keypoint_abs_path = os.path.join(keypoint_dir, keypoint)
            results = self.xmodel.fit(image_abs_path, keypoint_abs_path, gender)

            image_name = image.split('.')[0]
            mesh_results[image_name] = results
        
        return mesh_results