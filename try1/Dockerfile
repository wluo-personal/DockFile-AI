ARG cuda_version=9.0-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:${cuda_version}
ARG password_root='chenwei$123'
ARG user_name='kai'
ARG anaconda_version='Anaconda3-5.3.0-Linux-x86_64.sh'
RUN apt-get update && echo "root:${password_root}" | chpasswd \ 
    && adduser ${user_name} --gecos "Chen Wei,500,12345,12345" --disabled-password \
    && echo "${user_name}:${password_root}" | chpasswd \
    && apt-get install -y sudo \
        bzip2 \
        curl \
        vim \
    && adduser kai sudo 


# install anaconda
USER ${user_name}
RUN  cd ~ && mkdir anaconda && cd anaconda \
    && curl -O https://repo.continuum.io/archive/${anaconda_version} \
    && bash ${anaconda_version} -b \
    && echo "export PATH=/home/${user_name}/anaconda3/bin:$PATH" >> ~/.bashrc \
    && export PATH=/home/${user_name}/anaconda3/bin:$PATH \
    && yes|conda install -c conda-forge jupyter_contrib_nbextensions
    
# # install important packages
USER root
RUN apt-get update \
    && apt-get install -y git-core \
    && git config --global user.name "kai" \
    && git config --global user.email "kai@katabat.com"

# Create python CPU version
# In bashrc, it will return at the beginning unless it is running at interactive mode! source cmd will not work in docker build!
ARG python_cpu_version='3.6'
USER ${user_name}
WORKDIR /home/${user_name}
# link bash with /bin/sh. RUN command default to use /bin/sh instead of /bin/bash
RUN ["/bin/bash", "-c", "echo $password_root | sudo -S ln -sf bash /bin/sh"]
RUN export PATH=/home/${user_name}/anaconda3/bin:$PATH \
    && yes | conda create -n python${python_cpu_version}-CPU anaconda python=${python_cpu_version} \
    && source activate python${python_cpu_version}-CPU \
    && pip install --upgrade pip \
    && yes | pip install lightgbm \
    && yes | pip install xgboost \
    && yes | pip install catboost \
    && yes | pip install --upgrade tensorflow \
    && yes | pip install keras \
    && conda install ipykernel \
    && python -m ipykernel install --user --name python${python_cpu_version}-CPU --display-name "python${python_cpu_version}-CPU" \
    && source deactivate
    
    
# Create python GPU version
ARG python_gpu_version='3.6'
USER ${user_name}
RUN export PATH=/home/${user_name}/anaconda3/bin:$PATH \
    && yes | conda create -n python${python_gpu_version}-GPU anaconda python=${python_gpu_version} \
    && source activate python${python_gpu_version}-GPU \
    && pip install --upgrade pip \
    && conda install ipykernel \
    && python -m ipykernel install --user --name python${python_cpu_version}-GPU --display-name "python${python_gpu_version}-GPU" \
    && source deactivate
    
# lightgbm - GPU
RUN echo $password_root | sudo -S apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        libboost-dev  \
        libboost-system-dev \
        libboost-filesystem-dev \
    && cd /home/${user_name} && mkdir lightgbm && cd lightgbm \
    && git clone --recursive https://github.com/Microsoft/LightGBM \
    && cd LightGBM && mkdir build ; cd build \
    && cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. \
    && make -j$(nproc) \
    && cd .. && cd python-package/ \
    && export PATH=/home/${user_name}/anaconda3/bin:$PATH \
    && source activate python${python_gpu_version}-GPU \
    && python setup.py install --precompile \
    && cd .. \
    && pip install --upgrade tensorflow-gpu \
    && pip install keras \
    && source deactivate \
    && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64" >> /home/${user_name}/.bashrc
    
# solve opencl not found issue
USER root
RUN mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd



#setup jupyter notebook configuration
USER ${user_name}
COPY jupyter_notebook_config.py /home/${user_name}
RUN export PATH=/home/${user_name}/anaconda3/bin:$PATH \
    &&cd ~ && mkdir jupyter_certs && cd jupyter_certs \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -subj "/C=US/ST=Delaware/L=Wilmington/O=Katabat/OU=AI/CN=kai/emailAddress=kai@katabat.com" \
    -keyout mykey.key -out mycert.pem \
    && jupyter notebook --generate-config \
    && mv ~/jupyter_notebook_config.py ~/.jupyter
    
    



# link bash with /bin/sh. RUN command default to use /bin/sh instead of /bin/bash
# RUN ["/bin/bash", "-c", "echo $password_root | sudo -S ln -sf bash /bin/sh"]
# revoke bash with /bin/sh
RUN ["/bin/bash", "-c", "echo $password_root | sudo -S ln -sf dash /bin/sh"]

# run with sudo priviledge
# echo $password_root | sudo -S <cmd>
# echo $password_root | sudo -S -H -u <user> <cmd>
# link bash with /bin/sh 
# sudo ln -sf bash /bin/sh
# revert system to default
# sudo ln -sf dash /bin/sh
CMD ["/bin/bash"]

