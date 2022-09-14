# 3D FLAME Head model in SMPL-X format

## Installation

- Create a conda environment with python 3.7
```bash
conda create -n 3d_flame_head python=3.7.*
conda activate 3d_flame_head
```

- Install the requirements
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

- Install the enviroment for 3DDFA_v2
```bash
cd src/V2_3DDFA/FaceBoxes/utils && python build.py build_ext --inplace && cd -
```

- (Optional) Install the mesh renderer from [MPI-IS/mesh](https://github.com/MPI-IS/mesh)

- Download and extract SMPL-X model from [here](https://smpl-x.is.tue.mpg.de/) (models_smplx_v1_1.zip) and unzip it in the `models` folder. You should have a folder `models/smplx` with the following files:
```bash
./models/smplx/
            -SMPLX_FEMALE.npz
            -SMPLX_FEMALE.pkl
            -SMPLX_MALE.npz
            -SMPLX_MALE.pkl
            -SMPLX_NEUTRAL.npz
            -SMPLX_NEUTRAL.pkl
            -smplx_npz.zip
            -version.txt
```

- Download and extract 3DFFA models from [3DFFA_v2](https://github.com/cleardusk/3DDFA_V2) an unzip it in the `./src/V2_3DDFA/weights`. You should have a folder `./src/V2_3DDFA/weights` with the following files:
```bash
./src/V2_3DDFA/weights
                    -mb1
                        -----bfm_noneck_v3.onnx
                        -----FaceBoxesProd.onnx
                        -----mb1_120x120.onnx
                    -mb05
                        -----bfm_noneck_v3.onnx
                        -----FaceBoxesProd.onnx
                        -----mb05_120x120.onnx
                    -bfm_noneck_v3.pkl
                    -param_mean_std_62d_120x120.pkl
                    -tri.pkl
```
## Usage

```bash
python fit_2D_face_landmarks.py --input_image samples/nguyen.png
```