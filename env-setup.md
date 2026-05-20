
# How to pre-install environment for development

To install from scatch Jetson nano system image, please use the following instructions. Assume that username/password is superai. (This guide created by Thananop Kobchaisawat)


## 1. Append these in ~/.bashrc to make it persist environment variables
```
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:/usr/src/tensorrt/bin:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export CUDA_INC_DIR=/usr/local/cuda/include
export CPATH=$CUDA_HOME/lib64:$CPATH
export CUDA_ROOT=/usr/local/cuda
```


## 2. Install require system packages and monitoring tools
```
sudo apt install v4l-utils htop python3-setuptools python3-venv python3-pip python3-gi python3-gst-1.0 python3-numpy python3-opencv python3-dev build-esstential -y
sudo pip3 install -U jetson-stats #  jtop command
```

## 3. Install pycuda as python3 system-level package
```
sudo pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" -U pycuda
```

## 4. Create virtual environment for python and included system python packages into this env. THIS IS A MUST to PREVENT MESSY THINGS when trying to use H/W Acclerated compoents, otherwise you will need to compile the whole thing from source (GStreamer,OpenCV/TensorRT)
```
python3 -m venv --system-site-packages $HOME/labenv
```

## 5. Activate created env name labenv and install prebuilt onnxruntime-gpu
```
source  $HOME/labenv/bin/activate

#### This two commands will check that are you in correct env or not? If correct, it will be something like $HOME/labenv/bin/[pip|python]
which pip
which python
#### This two commands will check that are you in correct env or not? If correct, it will be something like $HOME/labenv/bin/[pip|python]

# https://elinux.org/Jetson_Zoo#ONNX_Runtime download onnxruntime from here python3.6!
pip install -U pip $HOME/packages/onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
```

## 6. Make venv activate when open shell. Append these to ~/.bashrc
```
source /home/superai/labenv/bin/activate
```
