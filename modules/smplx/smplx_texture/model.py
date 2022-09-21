from modules.smplx.model3d import BaseModel
import os
from modules.smplx.smplx_texture import textured_smplx
import shutil

ABS_DIR_PATH = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

class SMPLX_Texture(BaseModel):
    """SMPL-X head model"""
    def __init__(self, 
                model='smplx', # Choose (smpl, smplx)
                name='SMPLX_Texture'): # Add your aguments here
        super(SMPLX_Texture, self).__init__(name)
        self.model = model
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

    # the inherited predict function is used to call your custom functions
    def predict(self, smplifyx_out_path, front_img_path, back_img_path, output_fn, **kwargs):
        # step.0: check all the input data
    
        front_img = os.path.split(os.path.basename(front_img_path))[-1]
        back_img = os.path.split(os.path.basename(back_img_path))[-1]
        
        tmp = front_img.rfind('.')
        front_id = front_img[:tmp]
        tmp = back_img.rfind('.')
        back_id = back_img[:tmp]
        
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
        f_obj = os.path.join(smplifyx_out_path, 'meshes', front_id, '000.obj')
        f_pkl = os.path.join(smplifyx_out_path, 'results', front_id, '000.pkl')
        for fname, ftype in zip([f_img, f_obj, f_pkl], ['image','obj','pkl']):
            if not os.path.isfile(fname):
                raise(Exception("%d file for the front is not found"%ftype))
                
        b_img = back_img_path
        b_obj = os.path.join(smplifyx_out_path, 'meshes', back_id, '000.obj')
        b_pkl = os.path.join(smplifyx_out_path, 'results', back_id, '000.pkl')

        for fname, ftype in zip([b_img, b_obj, b_pkl], ['image','obj','pkl']):
            if not os.path.isfile(fname):
                raise(Exception("%d file for the back is not found"%ftype))
                
                
        npath = os.path.join(OUTPUT_DIR, output_fn)
            
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
        
        textured_smplx.complete_texture(f_acc_texture, f_acc_vis, f_mask)
        
        # finish: copy the result
        print('Copying the result...')
        shutil.copyfile(f_acc_texture[:-4]+'complete.png',
                        os.path.join(OUTPUT_DIR, '%s.png'%output_fn))

        return os.path.join(OUTPUT_DIR, '%s.png'%output_fn) 