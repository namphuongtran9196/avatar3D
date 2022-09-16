from django.http import HttpResponse
import os


from modules.smplx.smplx_pose.model import SMPLX_Pose

# Init models
# model_head = SMPLX_Head()
inputs = ['/Users/jaydentran1909/Documents/smplx/avatar3d/modules/smplx/smplx_pose/data/images/007.jpg']
model_pose = SMPLX_Pose()


def index(request):
    a = model_pose.predict(inputs)
    print(a)
    return HttpResponse("Test model finished! Please check the console output.")