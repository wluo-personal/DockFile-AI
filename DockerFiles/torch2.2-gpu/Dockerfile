# The official tf installation guidance does not work
# Once the cuda and cudnn is correctly configured. pip install tensorflow will
# install GPU env.

ARG cuda_version=12.1.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:${cuda_version}
ARG password_root='tomcat123'
ARG user_name='wei'
ARG conda_version='Miniconda3-latest-Linux-x86_64.sh'
ARG gpu_env_name='tc-gpu'

RUN apt-get update && echo "root:${password_root}" | chpasswd \
    && adduser ${user_name} --gecos "Wei Luo,500,12345,12345" --disabled-password \
    && echo "${user_name}:${password_root}" | chpasswd \
    && apt-get install -y sudo bzip2 curl vim wget htop \
    && adduser ${user_name} sudo \
    # opencv depends on below
    && apt-get update \
    # cv2 dependencies
    && DEBIAN_FRONTEND=noninteractive apt-get -yq install libgl1 libglib2.0-0


# install anaconda
USER ${user_name}
RUN  cd ~ && mkdir miniconda && cd miniconda \
    && curl https://repo.anaconda.com/miniconda/${conda_version} -o ${conda_version} \
    && bash ${conda_version} -b \
    && export PATH=/home/${user_name}/miniconda3/bin:$PATH \
    && conda init \
    && pip install jupyter_contrib_nbextensions \
    && yes | conda create --name ${gpu_env_name} python=3.11 --no-default-packages \
    && cd /home/${user_name} && rm -rf miniconda
    # jupyter notebook config \

 # config jupyter \
USER ${user_name}
RUN export PATH=/home/${user_name}/miniconda3/bin:$PATH \
    &&cd ~ && mkdir jupyter_certs && cd jupyter_certs \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -subj "/C=US/ST=NewJersey/L=Edgewater/O=Wei/OU=AI/CN=wei/emailAddress=luoweiforever@gmail.com" \
    -keyout mykey.key -out mycert.pem \
    && mkdir ~/.jupyter \
    && mkdir ~/.jupyter/nbconfig \
    && mkdir ~/.jupyter/custom \
    && jupyter notebook --generate-config



COPY ./jupyter/jupyter_notebook_config.py /home/${user_name}/.jupyter
#COPY ./jupyter/nbconfig /home/${user_name}/.jupyter/nbconfig
COPY ./jupyter/custom /home/${user_name}/.jupyter/custom

USER ${user_name}
RUN export PATH=/home/${user_name}/miniconda3/bin:$PATH \
    && conda run --name base conda install -c conda-forge -y jupyter_contrib_nbextensions \
    && conda run --name base pip install --upgrade nbconvert \
    && conda run --name base conda upgrade -y nbconvert

USER ${user_name}
RUN export PATH=/home/${user_name}/miniconda3/bin:$PATH \
    && conda run --name ${gpu_env_name} pip install ipykernel \
    && conda run --name ${gpu_env_name} python -m ipykernel install --user --name ${gpu_env_name} --display-name "${gpu_env_name}"


USER ${user_name}
#https://pytorch.org/get-started/previous-versions/
RUN export PATH=/home/${user_name}/miniconda3/bin:$PATH \
    && conda run --name ${gpu_env_name} pip install --upgrade pip \
    && conda run --name ${gpu_env_name} pip install poetry


USER root
RUN apt-get update && apt-get clean \
    && chown -R ${user_name} /home/${user_name}/.jupyter

USER ${user_name}
WORKDIR /home/${user_name}/data
CMD ["/bin/bash"]

