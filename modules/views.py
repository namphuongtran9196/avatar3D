from django.http import HttpResponse
from modules.smplx.smplx_head.model import SMPLX_Head
from modules.smplx.smplx_pose.model import SMPLX_Pose
from modules.smplx.smplx_texture.model import SMPLX_Texture
from modules.smplx.smplx_uvmap.model import SMPLX_UVmap

# Init models
# model_head = SMPLX_Head()
inputs = ['/Users/jaydentran1909/Documents/smplx/avatar3d/modules/smplx/smplx_pose/data/images/007.jpg']
model_pose = SMPLX_Pose()
# model_texture = SMPLX_Texture()
# model_uvmap = SMPLX_UVmap()


def index(request):
    a = model_pose.predict(inputs)
    print(a)
    return HttpResponse("Test model finished! Please check the console output.")