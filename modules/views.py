from django.http import HttpResponse
from modules.smplx.smplx_pose.model import SMPLX_Pose

# Init models
model = SMPLX_Pose()

def index(request):
    dir_name = 'test_upload'
    # For testing this, upload test images to `test_upload` folder
    # if there's no `test_upload` folder, please create folders `data/images` inside `smplx_pose`
    model.predict(dir_name, gender='male')
    return HttpResponse("Test model finished! Please check the console output.")