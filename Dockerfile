FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# ARG USERNAME=terry
ARG WORKDIR=/workspace

# RUN useradd terry
RUN apt-get update
RUN apt-get install -y vim ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 cython \
        python-dev python3-pip gcc g++ zip unzip curl zlib1g-dev pkg-config python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc cmake make libncursesw5-dev libgdbm-dev libc6-dev zlib1g-dev libsqlite3-dev libssl-dev openssl libffi-dev wget \
        sudo build-essential openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev libtbb2 libtbb-dev libopenblas-dev libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pip --upgrade
RUN pip3 install opencv-python h5py onnx onnxruntime-gpu onnx-simplifier timm tqdm pycuda

COPY TensorRT-7.2.2.3 /TensorRT-7.2.2.3
# RUN cd TensorRT-7.2.2.3/python && pip3 install tensorrt-7.2.2.3-cp37-none-linux_x86_64.whl
# RUN cd TensorRT-7.2.2.3/uff && pip3 install uff-0.6.5-py2.py3-none-any.whl
# RUN cd TensorRT-7.2.2.3/graphsurgeon && pip3 install graphsurgeon=0.4.1-py2.py3-none-any.whl
# ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64:/usr/local/cuda/extras/CUPTI/lib64:/TensorRT-7.2.2.3/lib"

# USER ${USERNAME}
# RUN sudo chown -R ${USERNAME}:${USERNAME} ${WORKDIR}
WORKDIR ${WORKDIR}
