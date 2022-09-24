# avatar3d
3D avatar from single image for metaverse

## Installation


### Download models
- Download and extract openpose model from [OpenPose](https://drive.google.com/file/d/1vGZ9FVkoK0D42LOsglyw-Oac4MNe7Sws/view?usp=sharing) and unzip it in the `modules/smplx/smplx_pose/openpose` or use the script to download model `getModels.sh`. You should have a folder `modules/smplx/smplx_pose/openpose/models` with the following files:
```bash
modules/smplx/smplx_pose/openpose/models
                                        ├──cameraParameters
                                                ├──flir
                                                        ├──17012332.xml.example
                                        ├──face
                                                ├──pose_deploy.prototxt
                                                ├──pose_iter_116000.caffemodel
                                        ├──hand
                                                ├──pose_deploy.prototxt
                                                ├──pose_iter_102000.caffemodel
                                        ├──pose
                                                ├──body_25
                                                        ├──pose_deploy.prototxt
                                                        ├──pose_iter_584000.caffemodel
                                                ├──coco
                                                        ├──pose_deploy_linevec.prototxt
                                                        ├──pose_iter_440000.caffemodel
                                                ├──mpi
                                                        ├──pose_deploy_linevec_faster_4_stages.prototxt
                                                        ├──pose_deploy_linevec.prototxt
                                                        ├──pose_iter_160000.caffemodel
                                        ├──getModels.bat
                                        ├──getModels.sh
```

- Download and extract smplx, vposer model from [SMPLX, VPoser](https://drive.google.com/file/d/1Q-azs3V8i88Td28fXL-4VGmgmJ_UL-58/view?usp=sharing) and unzip it in the `modules/smplx/smplx_pose/smplifyx`. You should have a folder `modules/smplx/smplx_pose/smplifyx/models` with the following files:
```bash
modules/smplx/smplx_pose/smplifyx/models
                                        ├──smplx
                                                ├──SMPLX_FEMALE.npz
                                                ├──SMPLX_FEMALE.pkl
                                                ├──SMPLX_MALE.npz
                                                ├──SMPLX_MALE.pkl
                                                ├──SMPLX_NEUTRAL.npz
                                                ├──SMPLX_NEUTRAL.pkl
                                                ├──smplx_npz.zip
                                                ├──version.txt
                                        ├──vposer_v1_0
                                                ├──snapshots
                                                             ├──TR00_E096.pt
                                                ├──TR00_004_00_WO_accad.ini
                                                ├──version.txt
                                                ├──vposer_smpl.py
```

- Download and extract 3DFFA models from [3DFFA_v2](https://drive.google.com/file/d/1VA0GMk2e2DSYNg1YeW4mmLI1uL_Z_d8T/view?usp=sharing) and unzip it in the `modules/smplx/smplx_head/src/V2_3DDFA`. You should have a folder `modules/smplx/smplx_head/src/V2_3DDFA/weights` with the following files:
```bash
modules/smplx/smplx_head/src/V2_3DDFA/weights
                                            ├──mb1
                                                    ├──bfm_noneck_v3.onnx
                                                    ├──FaceBoxesProd.onnx
                                                    ├──mb1_120x120.onnx
                                            ├──mb05
                                                    ├──bfm_noneck_v3.onnx
                                                    ├──FaceBoxesProd.onnx
                                                    ├──mb05_120x120.onnx
                                            ├──bfm_noneck_v3.pkl
                                            ├──param_mean_std_62d_120x120.pkl
                                            ├──tri.pkl
```

### Install dependencies
- Create a conda environment with python 3.7
```bash
conda env create -f enviroment.yml
conda activate avatar3d
pip install git+https://github.com/nghorbani/configer
```

- Install Cmake
```bash
wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz

tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local

rm cmake-3.13.0-Linux-x86_64.tar.gz
```

- Install the enviroment for 3DDFA_v2
```bash
cd modules/smplx/smplx_head/src/V2_3DDFA/FaceBoxes/utils && python build.py build_ext --inplace && cd -
```

- Install the enviroment for openpose and smplifyx
```bash
sudo apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev libboost-thread-dev

cd modules/smplx/smplx_pose/smplx && python setup.py install && cd -
```

- Build library from souce
```bash
cd modules/smplx/smplx_pose && sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt

# Compile openpose with CMakelists.txt and CPU_ONLY
cd openpose && rm -rf build || true && mkdir build && cd build 
cmake .. -DGPU_MODE=CPU_ONLY 
make -j`nproc` 
cd ../../../../..
```

- Fix the bug of torchgeometry
```bash
cp modules/smplx/smplx_pose/conversions.py path/to/your_anaconda3/envs/avatar3d/lib/python3.7/site-packages/torchgeometry/core/conversions.py
```

## Usage
- Run the server
```bash
python manage.py runserver
```
```
- Test the server api
```bash
python post_request.py -f path/to/image_front.jpg -b path/to/image_back.jpg
```
- The output obj file and uvmap will be written to `modules/outputs`
