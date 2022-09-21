from django.http import HttpResponse
from modules.smplx.smplx_texture.model import SMPLX_Texture

model = SMPLX_Texture()

def index(request):
    dir_name = 'test_upload'
    smplifyx_out_path = '/Users/jaydentran1909/Documents/avatar3d/modules/smplx/smplx_pose/data/smplifyx_results'
    front_img_path = '/Users/jaydentran1909/Documents/avatar3d/modules/smplx/smplx_pose/data/images/005_front.jpg'
    back_img_path = '/Users/jaydentran1909/Documents/avatar3d/modules/smplx/smplx_pose/data/images/005_back.jpg'
    model.predict(smplifyx_out_path, front_img_path, back_img_path, dir_name)
    return HttpResponse("Test model finished! Please check the console output.")