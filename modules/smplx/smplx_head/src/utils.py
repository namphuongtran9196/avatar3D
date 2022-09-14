
import cv2
import os
import numpy as np
import torch

def torch_project_points(points, scale, trans, device):
    '''
    weak perspective camera
    project 3D points onto 2D image
    '''
    return torch.mul(scale, torch.transpose(torch.matmul(torch.eye(n=2, m=3, dtype=points.dtype).to(device), points.T), 1, 0)+ trans)

def save_npy(output_path, global_orient, jaw_pose, leye_pose, reye_pose, expression, body_pose, transl, scale, dummy_input):
    '''
    save the optimized parameters
    '''
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path + '/global_orient.npy', global_orient)
    np.save(output_path + '/jaw_pose.npy', jaw_pose)
    np.save(output_path + '/leye_pose.npy', leye_pose)
    np.save(output_path + '/reye_pose.npy', reye_pose)
    np.save(output_path + '/expression.npy', expression)
    np.save(output_path + '/body_pose.npy', body_pose)
    np.save(output_path + '/transl.npy', transl)
    np.save(output_path + '/scale.npy', scale)
    np.save(output_path + '/dummy_input.npy', dummy_input)
    
def load_npy(init_params):
    '''
    load the initial parameters
    '''
    global_orient = np.load(init_params + '/global_orient.npy')
    jaw_pose = np.load(init_params + '/jaw_pose.npy')
    leye_pose = np.load(init_params + '/leye_pose.npy')
    reye_pose = np.load(init_params + '/reye_pose.npy')
    expression = np.load(init_params + '/expression.npy')
    body_pose = np.load(init_params + '/body_pose.npy')
    transl = np.load(init_params + '/transl.npy')
    scale = np.load(init_params + '/scale.npy')
    dummy_input = np.load(init_params + '/dummy_input.npy')

    return global_orient, jaw_pose, leye_pose, reye_pose, expression, body_pose, transl, scale, dummy_input


def load_uvmap(uvmap_path):
    with open(uvmap_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    f = []
    fv = []
    ft = []
    fn = []
    vt = []
    vn = []
    for line in lines:
        if line.startswith('f '):
            f.append(line)
        elif line.startswith('vt '):
            vt.append((float(line.split()[1]), 1.0 - float(line.split()[2])))
        elif line.startswith('vn '):
            vn.append((float(line.split()[1]), float(line.split()[2]), float(line.split()[3])))

    for l in f:
        fv.append([int(l.split()[1].split('/')[0]) - 1,int(l.split()[2].split('/')[0]) - 1, int(l.split()[3].split('/')[0]) - 1])
        ft.append([int(l.split()[1].split('/')[1]) - 1,int(l.split()[2].split('/')[1]) - 1,int(l.split()[3].split('/')[1]) - 1])
        fn.append([int(l.split()[1].split('/')[2]) - 1,int(l.split()[2].split('/')[2]) - 1,int(l.split()[3].split('/')[2]) - 1])
        
    return np.asarray(vt), np.asarray(vn), np.asarray(fv), np.asarray(ft), np.asarray(fn)

def transfer_single_trimesh_texture(src, dst, tri1, tri2):

    # Find bounding box. 
    r1 = cv2.boundingRect(tri1) 
    r2 = cv2.boundingRect(tri2)
    
    # Offset points by left top corner of the respective rectangles
    tri1Cropped = []
    tri2Cropped = []
        
    for i in range(0, 3):
        tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
        tri2Cropped.append(((tri2[0][i][0] - r2[0]),(tri2[0][i][1] - r2[1])))

    # Crop input image
    img1Cropped = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
    # Apply the Affine Transform just found to the src image
    img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);

    img2Cropped = img2Cropped * mask
        
    # Copy triangular region of the rectangular patch to the output image
    dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

    return dst