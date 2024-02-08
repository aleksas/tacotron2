FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS dev
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN python -m pip install --upgrade pip \
    && pip install packaging cython future jupyter
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y git llvm pkg-config wget gfortran
RUN git clone https://github.com/nvidia/apex.git  /root/workspaces/apex
WORKDIR /root/workspaces/apex
RUN git reset --hard 7b2e71b0d4013f8e2f9f1c8dd21980ff1d76f1b6
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" /root/workspaces/apex

RUN mkdir -p /root/workspaces/
COPY ./requirements.txt /root/workspaces/tacotron2/requirements.txt
RUN pip install -r /root/workspaces/tacotron2/requirements.txt

WORKDIR /root/workspaces/tacotron2

FROM dev AS build
