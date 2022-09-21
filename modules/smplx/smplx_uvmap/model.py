import sys
sys.path.insert(1, 'avatar3d\\modules\\smplx')

from model3d import BaseModel
import numpy as np
import cv2

class SMPLX_UVmap(BaseModel):
    """SMPL-X uv-map model"""
    def __init__(self, name='SMPLX_UVmap'): # Add your arguments here
        super(SMPLX_UVmap, self).__init__(name)

    # the inherited predict function is used to call your custom functions
    def predict(self, uv, uv_head, size=(1000, 1000)):
        """
        It takes in the UV map of the SMPLX model and the UV map of the head model, and returns the UV
        map of the SMPLX model with the head model's UV map pasted on top of it
        
        :param uv: the UV map of the SMPLX model
        :param uv_head: The UV map of the head
        :param size: the size of the output image
        :return: The uvmap is being returned.
        """
 
        print("Predict from SMPLX_UVmap")
        uvmap = np.where(uv_head==0, uv, uv_head)
        return cv2.resize(uvmap, size)

             
if __name__ == '__main__':
    uv = cv2.imread('D:\\2D_to_3D\\texture_smplx.png')
    uv_head = cv2.imread('D:\\2D_to_3D\\texture_smplx2.png')
    smplx_uvmap = SMPLX_UVmap()
    result = smplx_uvmap.predict(uv, uv_head)
    cv2.imshow('Result', result)
    cv2.waitKey(0)