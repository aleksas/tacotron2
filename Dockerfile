FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS dev
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN python -m pip install --upgrade pip
RUN pip install packaging cython
RUN apt update \
    && apt install -y git llvm
#RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@e87b5799e697c5ee45119e669d37bb06f6b975ac
RUN DEBIAN_FRONTEND=noninteractive apt -y install pkg-config wget gfortran
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
#RUN pip install apex==0.9.10.dev0 
RUN pip install future

FROM dev AS build


RUN pip install future

# RUN mkdir -p /data \
#     && wget -nc https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -P /data/
# RUN if [ ! -d "data/LJSpeech-1.1" ]; then tar -xvf  data/LJSpeech-1.1.tar.bz2 -C data; fi

#RUN sed -i -- 's,DUMMY,data/LJSpeech-1.1/wavs,g' filelists/*.txt
#RUN python train.py --output_directory=outdir --log_directory=logdir
