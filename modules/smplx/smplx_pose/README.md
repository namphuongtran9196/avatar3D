# SMPLX-Pose API
The API for estimate 3D mesh from 2D single image using OpenPose and SMPLX model.

## Installation (for Linux only)
### Dependencies
```
# for backend
pip install django

# cmake
wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz

tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local

rm cmake-3.13.0-Linux-x86_64.tar.gz

# Others
apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev

pip install chumpy
pip install smplx
cd smplx && python setup.py install && cd ..
pip install git+https://github.com/nghorbani/configer
pip install human_body_prior
pip install torch==1.1.0
pip install -r smplifyx/requirements.txt
```

### Pretrained models
#### OpenPose models
Change directory to the `openpose/models` folder and run following shell:
```
bash getModels.sh
```
#### SMPLX models
Change directory to the `smplifyx` and run following shell:
```
mkdir models
gdown --folder https://drive.google.com/drive/folders/1ZzFrudqo2BvvxJmGKE2qqsIXU5i2Kz4I?usp=sharing -O models
```

### Build packages
Change directory back to the `smplx_pose` folder:
```
!sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt

cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. -DGPU_MODE=CPU_ONLY && make -j`nproc` && cd ..
```


## Usage
```python
python manage.py runserver
```
or
```python
from modules.smplx.smplx_pose.model import SMPLX_Pose

model = SMPLX_Pose(model_type='smplx')

model.predict(...) # Check source code for more detailed parameters
```