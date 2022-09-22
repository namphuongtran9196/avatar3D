import os
from modules.smplx.model3d import BaseModel
from modules.smplx.smplx_texture import textured_smplx

ABS_DIR_PATH = os.path.dirname(__file__)

class SMPLX_Texture(BaseModel):
    """SMPL-X head model"""
    def __init__(self, 
                model='smplx', # Choose (smpl, smplx)
                name='SMPLX_Texture',
                ): # Add your aguments here
        super(SMPLX_Texture, self).__init__(name)
        self.model = model

    # the inherited predict function is used to call your custom functions
    def predict(self, smplifyx_out_path, front_img_path, back_img_path, output_fn, **kwargs):
        # step.0: check all the input data
        
        front_id = 'front'
        back_id = 'back'
        
        if self.model == 'smpl':
            template_obj = os.path.join(ABS_DIR_PATH, 'models/smpl_uv.obj')
            template_mask = os.path.join(ABS_DIR_PATH,'models/smpl_mask_1000.png')
        elif self.model == 'smplx':
            template_obj = os.path.join(ABS_DIR_PATH,'models/smplx_uv.obj')
            template_mask = os.path.join(ABS_DIR_PATH,'models/smplx_mask_1000.png')
        else:
            raise(Exception("model type not found"))
            
        if not os.path.isfile(template_obj) or not os.path.isfile(template_mask):
            raise(Exception("model not found"))
            
        f_img = front_img_path
        f_obj = os.path.join(smplifyx_out_path, 'meshes', front_id, 'front.obj')
        f_pkl = os.path.join(smplifyx_out_path, 'results', front_id, 'front.pkl')

        for fname, ftype in zip([f_img, f_obj, f_pkl], ['image','obj','pkl']):
            if not os.path.isfile(fname):
                raise(Exception("%d file for the front is not found"%ftype))
                
        b_img = back_img_path
        b_obj = os.path.join(smplifyx_out_path, 'meshes', back_id, 'back.obj')
        b_pkl = os.path.join(smplifyx_out_path, 'results', back_id, 'back.pkl')

        for fname, ftype in zip([b_img, b_obj, b_pkl], ['image','obj','pkl']):
            if not os.path.isfile(fname):
                raise(Exception("%d file for the back is not found"%ftype))
                
                
        npath = os.path.join(smplifyx_out_path, output_fn)
            
        # step.1: produce single frame texture
        print('Producing single frame texture')    
        textured_smplx.get_texture_SMPL(f_img, f_obj, f_pkl, npath, 'front', template_obj) 
        textured_smplx.get_texture_SMPL(b_img, b_obj, b_pkl, npath, 'back', template_obj)   
        
        # step.2: produce PGN texture (optional)
        
        # textured_smplx.get_texture_SMPL(f_pgn, f_obj, f_pkl, npath, 'front_PGN', template_obj)
        # textured_smplx.get_texture_SMPL(b_pgn, b_obj, b_pkl, npath, 'back_PGN', template_obj)
        
        # step3: combine all the textures
        print('Combine textures...')
        textured_smplx.combine_texture_SMPL(npath)
        
        # step4: complete all the textures
        print('Completing all the textures...')
        f_acc_texture = os.path.join(npath, 'back_texture_acc.png')
        f_acc_vis = os.path.join(npath, 'back_texture_vis_acc.png')
        f_mask = template_mask
        
        complete_uvmap = textured_smplx.complete_texture(f_acc_texture, f_acc_vis, f_mask)
        
        return complete_uvmap