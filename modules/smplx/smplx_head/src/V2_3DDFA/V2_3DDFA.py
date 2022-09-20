# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
import yaml
from .FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from .TDDFA_ONNX import TDDFA_ONNX

class Face3DDFA(object):
    def __init__(self, config_path):
        cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
        self.face_boxes = FaceBoxes_ONNX(cfg)
        self.tddfa = TDDFA_ONNX(**cfg)

    def predict(self,img):
        """ Face bbox detection and 3DMM regression

        Args:
            img (numpy array): 3D image in BGR format and uint8 type (H,W,C)
        
        Returns:
            numpy array: the list of bounding boxes and landmarks.
        """
        # face detection
        bboxes = self.face_boxes(img)
        # regress 3DMM params
        param_lst, roi_box_lst = self.tddfa(img, bboxes)
        # reconstruct vertices landmarks
        landmarks_3D = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
        
        return bboxes, landmarks_3D