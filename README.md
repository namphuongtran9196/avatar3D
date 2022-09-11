# avatar3d
3D avatar from single image for metaverse

# Installation
```bash
pip install django
```

# Usage
- Please define your model in the folder modules/smplx/MODEL_NAME/
- You need to create the ```predict``` function in ```model.py``` and return the output required.
```bash
For smplx_head -> return:
                        jaw_pose # numpy array [Batch_size, 3]
                        leye_pose # numpy array [Batch_size, 3]
                        reye_pose # numpy array [Batch_size, 3]
                        expression # numpy array [Batch_size, num_expression_coeffs]
                        uvmap # an uvmap with size [1000,1000,3] (flame head smplx uvmap)
FOr smplx_pose -> return:
                        global_orient # numpy array [Batch_size, 3]
                        body_pose # numpy array [Batch_size, num_body_joints * 3]
                        betas # numpy array [Batch_size, num_betas]
                        left_hand_pose # numpy array [Batch_size, num_pca_comps]
                        right_hand_pose # numpy array [Batch_size, num_pca_comps]

For smplx_texture -> return:
                        uvmap # an uvmap with size [1000,1000,3] (completed smplx uvmap)
For smplx_uvmap -> return:
                        uvmap # an uvmap with size [1000,1000,3] (the merged uvmap)
```
```bash
python manage.py runserver
```