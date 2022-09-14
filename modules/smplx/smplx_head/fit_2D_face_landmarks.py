import os
import cv2
import torch
import tqdm
import argparse
import numpy as np
from src.models import SMPLX
from src.V2_3DDFA.V2_3DDFA import Face3DDFA
from torch.autograd import Variable
from src.utils import *

def main(args):
    if args.visualize_3D:
        from psbody.mesh import Mesh
        from psbody.mesh.meshviewer import MeshViewers

    if os.path.exists(os.path.splitext(args.input_image)[0]) and args.init_params is None:
        print('The output path already exists. Please delete it first. {}'.format(os.path.splitext(args.input_image)[0]))
        return
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    img_raw = cv2.imread(args.input_image)
    face_3ddfa = Face3DDFA("./src/V2_3DDFA/configs/mb1_120x120.yml")
    
    # Predict landmarks and face bounding box
    bboxes, landmarks_3D = face_3ddfa.predict(img_raw)
    # Crop the face
    bbox = bboxes[0]
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w_ex, h_ex = w * 0.2, h * 0.2
    img_crop = img_raw[int(bbox[1] - h_ex):int(bbox[3] + h_ex), int(bbox[0] - w_ex):int(bbox[2] + w_ex)]
    source_img = img_crop.copy()
    lmk3D = landmarks_3D[0].T
    lmk3D[:, 0] = lmk3D[:, 0] - bbox[0] + w_ex
    lmk3D[:, 1] = lmk3D[:, 1] - bbox[1] + h_ex

    lmk3D_true = np.zeros_like(lmk3D)
    if args.visualize:
        for p in lmk3D:
            cv2.circle(img_crop, (int(p[0]), int(p[1])), args.joints_size, (0, 0, 255), -1)

    # Change the landmarks order to fit the SMPL-X model
    lmk3D_true[:10, :] = lmk3D[17:27, :] # brow
    lmk3D_true[10:19, :] = lmk3D[27:36,:] # nose
    lmk3D_true[19:31, :] = lmk3D[36:48,:] # eye
    lmk3D_true[31:51, :] = lmk3D[48:68,:]# lip
    lmk3D_true[51:,:] = lmk3D[:17, :] # jaw

    # normalize
    lmk3D_true[:,1] = img_crop.shape[0] - lmk3D_true[:,1] # invert y axis
    lmk3D_true = lmk3D_true / np.array([img_crop.shape[1], img_crop.shape[0], np.max(lmk3D_true[:, 2])])
    lmk3D_true = torch.from_numpy(lmk3D_true).float().to(device)
    
    # The model gender
    gender = args.gender # 'neutral' or 'male' or 'female'
    # Whether to compute the keypoints that form the facial contour
    use_face_contour=True
    # Number of shape components to use
    num_betas = 100
    # Number of expression components to use
    num_expression_coeffs = 5000
    # The number of PCA components to use for each hand
    num_pca_comps = 6
                
    model = SMPLX(model_path = "models/smplx", 
                gender=gender, 
                use_face_contour=use_face_contour,
                num_betas=num_betas,
                num_expression_coeffs=num_expression_coeffs,
                num_pca_comps = num_pca_comps).to(device)
    for layer in model.parameters():
        layer.requires_grad = False
    
    # Control the model shape, like thin or fat. [Batch_size , num_betas]  
    betas = torch.zeros([1, model.num_betas], dtype=torch.float32).to(device)
    # Control the hand pose. [Batch_size, num_pca_comps]
    left_hand_pose = torch.zeros([1, model.num_pca_comps], dtype=torch.float32).to(device)
    right_hand_pose = torch.zeros([1, model.num_pca_comps], dtype=torch.float32).to(device)
    # Set dtype to float32
    dtype = torch.cuda.FloatTensor if args.use_gpu else torch.FloatTensor
    

    if args.init_params is not None and os.path.exists(args.init_params):
        global_orient, jaw_pose, leye_pose, reye_pose, expression, body_pose, transl, scale, dummy_input = load_npy(args.init_params)
        # Create dummy input data without gradients
        dummy_input = Variable(torch.from_numpy(dummy_input).type(dtype), requires_grad=False)
        # Create torch variables
        global_orient_variable = Variable(torch.from_numpy(global_orient).type(dtype), requires_grad=True) # [Batch_size, 3]
        jaw_pose_variable = Variable(torch.from_numpy(jaw_pose).type(dtype), requires_grad=True) # [Batch_size, 3]
        leye_pose_variable = Variable(torch.from_numpy(leye_pose).type(dtype), requires_grad=True)  # [Batch_size, 3] [x, y]
        reye_pose_variable = Variable(torch.from_numpy(reye_pose).type(dtype), requires_grad=True) # [Batch_size, 3]
        transl_variable = Variable(torch.from_numpy(transl).type(dtype), requires_grad=True) # [Batch_size, 3]
        expression_variable = Variable(torch.from_numpy(expression).type(dtype), requires_grad=True) # [Batch_size, num_expression_coeffs]
        body_pose_variable = Variable(torch.from_numpy(body_pose).type(dtype), requires_grad=True) # [Batch_size, num_body_joints, 3]
        torch_scale_variable = Variable(torch.from_numpy(scale).type(dtype), requires_grad=True)
    else:
        # Create dummy input data without gradients
        dummy_input = Variable(torch.randn(1, 1).type(dtype), requires_grad=False)
        # Create torch variables
        global_orient_variable = Variable(torch.zeros([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
        jaw_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
        leye_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True)  # [Batch_size, 3] [x, y]
        reye_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
        transl_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
        expression_variable = Variable(torch.randn([1, model.num_expression_coeffs]).type(dtype), requires_grad=True) # [Batch_size, num_expression_coeffs]
        body_pose_variable = Variable(torch.zeros([1, model.NUM_BODY_JOINTS * 3]).type(dtype), requires_grad=True) # [Batch_size, num_body_joints, 3]
        
        # Compute and print loss using operations on Variables.
        s2d = torch.mean(torch.norm(lmk3D_true[:,:2] -torch.mean(lmk3D_true[:,:2], dim=0), dim=1))
        
            
        global_orient_pred = dummy_input.mm(global_orient_variable)
        jaw_pose_pred = dummy_input.mm(jaw_pose_variable)
        leye_pose_pred = dummy_input.mm(leye_pose_variable)
        reye_pose_pred = dummy_input.mm(reye_pose_variable)
        expression_pred = dummy_input.mm(expression_variable)
        body_pose_pred = dummy_input.mm(body_pose_variable)
        transl_pred = dummy_input.mm(transl_variable)

        output = model(
                betas=betas,
                global_orient=global_orient_pred,
                expression=expression_pred,
                body_pose=body_pose_pred,
                left_hand_pose = left_hand_pose,
                right_hand_pose = right_hand_pose,
                jaw_pose = jaw_pose_pred,
                leye_pose = leye_pose_pred,
                reye_pose = reye_pose_pred,
                transl = transl_pred,
                    return_verts=True,)
        s3d = torch.mean(torch.sqrt(torch.sum(torch.square(output.joints.squeeze()[76:,:]-torch.mean(output.joints.squeeze()[76:,:], axis=0))[:, :2], dim=1)))    
        torch_scale_variable = Variable((s2d/s3d).type(dtype), requires_grad=True)

    
    # Optimize the parameters of the model
    learning_rate = 0.001
    best_loss = 999999
    loss = 99999
    t = 0
    while loss > 0.001:
        t+=1
        # Forward pass
        global_orient_pred = dummy_input.mm(global_orient_variable)
        jaw_pose_pred = dummy_input.mm(jaw_pose_variable)
        leye_pose_pred = dummy_input.mm(leye_pose_variable)
        reye_pose_pred = dummy_input.mm(reye_pose_variable)
        expression_pred = dummy_input.mm(expression_variable)
        body_pose_pred = dummy_input.mm(body_pose_variable)
        transl_pred = dummy_input.mm(transl_variable)

        body_pose_pred = torch.zeros([1, model.NUM_BODY_JOINTS * 3]).type(dtype)
        output = model(
                betas=betas,
                global_orient=global_orient_pred,
                expression=expression_pred,
                body_pose=body_pose_pred,
                left_hand_pose = left_hand_pose,
                right_hand_pose = right_hand_pose,
                jaw_pose = jaw_pose_pred,
                leye_pose = leye_pose_pred,
                reye_pose = reye_pose_pred,
                transl = transl_pred,
                   return_verts=True,)
    
        lmk3D_pred = output.joints.squeeze()[76:,:]

        lmks_proj_2d = torch_project_points(lmk3D_pred, torch_scale_variable, torch.zeros(2).to(device), device)

        loss = (lmks_proj_2d - lmk3D_true[:,:2]).pow(2).sum()
        
        if t % 5 == 0:
            print('\n',t, loss.detach().cpu().numpy(), end='')
        if np.isnan(loss.detach().cpu().numpy()):
            print("Reinitializing because of NaN")
            # Create dummy input data without gradients
            dummy_input = Variable(torch.randn(1, 1).type(dtype), requires_grad=False)
            # Create torch variables
            global_orient_variable = Variable(torch.zeros([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
            jaw_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
            leye_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True)  # [Batch_size, 3] [x, y]
            reye_pose_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
            transl_variable = Variable(torch.randn([1, 3]).type(dtype), requires_grad=True) # [Batch_size, 3]
            expression_variable = Variable(torch.randn([1, model.num_expression_coeffs]).type(dtype), requires_grad=True) # [Batch_size, num_expression_coeffs]
            body_pose_variable = Variable(torch.zeros([1, model.NUM_BODY_JOINTS * 3]).type(dtype), requires_grad=True) # [Batch_size, num_body_joints, 3]
            torch_scale_variable = Variable((s2d/s3d).type(dtype), requires_grad=True)
            loss = 99999999
            continue

        # Use autograd to compute the backward pass.
        loss.backward()

        # Update weights using gradient descent;
        global_orient_variable.data -= learning_rate * global_orient_variable.grad.data
        jaw_pose_variable.data -= learning_rate * jaw_pose_variable.grad.data
        leye_pose_variable.data -= learning_rate * leye_pose_variable.grad.data
        reye_pose_variable.data -= learning_rate * reye_pose_variable.grad.data
        expression_variable.data -= learning_rate * expression_variable.grad.data
        # body_pose_variable.data -= learning_rate * body_pose_variable.grad.data
        transl_variable.data -= learning_rate * transl_variable.grad.data
        torch_scale_variable.data -= learning_rate * torch_scale_variable.grad.data
        # Manually zero the gradients after updating weights
        global_orient_variable.grad.data.zero_()
        jaw_pose_variable.grad.data.zero_()
        leye_pose_variable.grad.data.zero_()
        reye_pose_variable.grad.data.zero_()
        expression_variable.grad.data.zero_()
        # body_pose_variable.grad.data.zero_()
        transl_variable.grad.data.zero_()
        torch_scale_variable.grad.data.zero_()

        # Visualize the results
        if args.visualize:

            img = img_crop.copy()
            try:
                for p in lmks_proj_2d.detach().cpu().numpy():
                    cv2.circle(img, (int(p[0]*img.shape[1]), img.shape[0] - int(p[1]*img.shape[0])), args.joints_size, (255, 0, 0), -1)
            except:
                pass
            cv2.imshow('img', cv2.resize(img, (640,640)))
            cv2.waitKey(1)
            if args.visualize_3D and t % 500 == 0:
                mv = MeshViewers(shape=[1,1], keepalive=True)
                mv[0][0].set_static_meshes([Mesh(output.vertices.detach().cpu().numpy()[0], model.faces)])
        
        if loss < best_loss and t % 5 == 0:
            print(" Save parameters at loss", loss.detach().cpu().numpy(), end='')
            best_loss = loss
            output_path = os.path.splitext(args.input_image)[0]
            save_npy(output_path, global_orient_variable.data.detach().cpu().numpy(), 
                                    jaw_pose_variable.data.detach().cpu().numpy(), 
                                    leye_pose_variable.data.detach().cpu().numpy(), 
                                    reye_pose_variable.data.detach().cpu().numpy(),
                                    expression_variable.data.detach().cpu().numpy(), 
                                    body_pose_variable.data.detach().cpu().numpy(), 
                                    transl_variable.data.detach().cpu().numpy(), 
                                    torch_scale_variable.data.detach().cpu().numpy(),
                                    dummy_input.data.detach().cpu().numpy())
        if t > args.max_iter:
            break
    print("\nFit 2D landmarks to 3D model and saved to: {} with the lowest loss {}".format(output_path, best_loss))

    print("Creating UV map")
    # Forward pass
    global_orient_pred = dummy_input.mm(global_orient_variable)
    jaw_pose_pred = dummy_input.mm(jaw_pose_variable)
    leye_pose_pred = dummy_input.mm(leye_pose_variable)
    reye_pose_pred = dummy_input.mm(reye_pose_variable)
    expression_pred = dummy_input.mm(expression_variable)
    body_pose_pred = dummy_input.mm(body_pose_variable)
    transl_pred = dummy_input.mm(transl_variable)
    body_pose_pred = torch.zeros([1, model.NUM_BODY_JOINTS * 3]).type(dtype)
    output = model(
            betas=betas,
            global_orient=global_orient_pred,
            expression=expression_pred,
            body_pose=body_pose_pred,
            left_hand_pose = left_hand_pose,
            right_hand_pose = right_hand_pose,
            jaw_pose = jaw_pose_pred,
            leye_pose = leye_pose_pred,
            reye_pose = reye_pose_pred,
            transl = transl_pred,
                return_verts=True,)

    proj_2d_points = torch_project_points(output.vertices[0], torch_scale_variable, torch.zeros(2), torch.device('cpu'))
    proj_2d_points[:, 1] = 1 - proj_2d_points[:, 1]
    
    
    proj_2d_points = proj_2d_points.detach().numpy().astype(np.float32)
    vt,vn , fv, ft, fn  = load_uvmap(args.uvmap_path)
    
    # load source image
    texture_img = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
    
    for i in tqdm.tqdm(range(len(fv))):
        # get vertices of the face
        v1, v2, v3 = fv[i]
        # get the corresponding uv index
        index_uv1, index_uv2, index_uv3 = ft[i]
        uv1, uv2, uv3 = vt[index_uv1], vt[index_uv2], vt[index_uv3]

        # get the corresponding normal index
        index_n1, index_n2, index_n3 = fn[i]
        n1, n2, n3 = vn[index_n1], vn[index_n2], vn[index_n3]
        
        if n1[2] > 0 and n2[2] > 0 and n3[2] > 0:
            # convert uv to pixel
            uv1 = [int(uv1[0] * args.img_size), int(uv1[1] * args.img_size)]
            uv2 = [int(uv2[0] * args.img_size), int(uv2[1] * args.img_size)]
            uv3 = [int(uv3[0] * args.img_size), int(uv3[1] * args.img_size)]
            
            # convert xy to pixel
            xy1 = [int(proj_2d_points[v1][0] * source_img.shape[1]), int(proj_2d_points[v1][1] * source_img.shape[0])]
            xy2 = [int(proj_2d_points[v2][0] * source_img.shape[1]), int(proj_2d_points[v2][1] * source_img.shape[0])]
            xy3 = [int(proj_2d_points[v3][0] * source_img.shape[1]), int(proj_2d_points[v3][1] * source_img.shape[0])]
            
            
            valid_point = True
            for x, y in [xy1, xy2, xy3]:
                if x < 0 or x >= source_img.shape[1] or y < 0 or y >= source_img.shape[0]:
                    valid_point = False
            if not valid_point:
                continue

            tri1 = np.float32([[xy1, xy2, xy3]])
            tri2 = np.float32([[uv1, uv2, uv3]])

            black_img = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)

            single_trimesh_texture = transfer_single_trimesh_texture(source_img, black_img, tri1, tri2)
        
            texture_img = np.where(single_trimesh_texture != 0, single_trimesh_texture, texture_img)
    
    cv2.imwrite(os.path.splitext(args.input_image)[0]+"_UVmap.png", texture_img)
    print("Saved UV map to {}".format(os.path.splitext(args.input_image)[0]+"_UVmap.png"))
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='./samples/nguyen.png',help="Path to the input image")
    parser.add_argument('--init_params', type=str, default=None, help="Optional. Path to the saved parameters to initialize the optimization")
    parser.add_argument('--uvmap_path', type=str, default='./src/uvmap.obj', help="Path to the uvmap obj")
    parser.add_argument('--img_size', type=int, default=1000, help="The output of uvmap image size")
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral','male', 'female'], help="The gender of model")
    parser.add_argument('--max_iter',type=int, default=10000, help='The maximum iteration for fitting landmarks')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU to speed up the fitting')
    parser.add_argument('--visualize', action='store_true', help='Visualize the fitting process')
    parser.add_argument('--visualize_3D', action='store_true', help='Visualize the 3D fitting process')
    parser.add_argument('--joints_size', type=int, default=1, help='The size of joints drawn on the image')

    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    main(args)