from django.http import HttpResponse
from modules.smplx.smplx_head.model import SMPLX_Head
from modules.smplx.smplx_pose.model import SMPLX_Pose
from modules.smplx.smplx_texture.model import SMPLX_Texture
from modules.smplx.smplx_uvmap.model import SMPLX_UVmap

# Init models
model_head = SMPLX_Head()
model_pose = SMPLX_Pose()
model_texture = SMPLX_Texture()
model_uvmap = SMPLX_UVmap()
    
def index(request):
    model_head.predict(None)
    model_pose.predict(None)
    model_texture.predict(None)
    model_uvmap.predict(None)
    return HttpResponse("Test model finished! Please check the console output.")