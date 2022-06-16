ARG cuda_version=11.2.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:${cuda_version}
ARG password_root='tomcat123'
ARG user_name='wei'
ARG conda_version='Miniconda3-latest-Linux-x86_64.sh'

RUN apt-get update && echo "root:${password_root}" | chpasswd \
    && adduser ${user_name} --gecos "Wei Luo,500,12345,12345" --disabled-password \
    && echo "${user_name}:${password_root}" | chpasswd \
    && apt-get install -y sudo \
        bzip2 \
        curl \
        vim \
        wget \
    && adduser ${user_name} sudo


# install anaconda
USER ${user_name}
RUN  cd ~ && mkdir miniconda && cd miniconda \
    && curl https://repo.anaconda.com/miniconda/${conda_version} -o ${conda_version} \
    && bash ${conda_version} -b \
    && export PATH=/home/${user_name}/miniconda3/bin:$PATH \
    && conda init \
    && pip install jupyter_contrib_nbextensions \
    && yes | conda create --name tf-gpu python=3.9 --no-default-packages \
    && pip install --upgrade pip \
    && pip install tensorflow-gpu==2.7 \
    && cd /home/${user_name} && rm -rf miniconda
    # jupyter notebook config \

 # config jupyter \
USER ${user_name} \
RUN export PATH=/home/${user_name}/miniconda3/bin:$PATH \
     && source base \


    &&cd ~ && mkdir jupyter_certs && cd jupyter_certs \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -subj "/C=US/ST=NewJersey/L=Edgewater/O=Wei/OU=AI/CN=wei/emailAddress=luoweiforever@gmail.com" \
    -keyout mykey.key -out mycert.pem \
    && jupyter notebook --generate-config
COPY jupyter_notebook_config.py /home/${user_name}/.jupyter/





USER root
RUN apt-get update && apt-get clean

# link bash with /bin/sh. RUN command default to use /bin/sh instead of /bin/bash
# RUN ["/bin/bash", "-c", "echo $password_root | sudo -S ln -sf bash /bin/sh"]
# revoke bash with /bin/sh
RUN ["/bin/bash", "-c", "echo $password_root | sudo -S ln -sf dash /bin/sh"]



USER ${user_name}
WORKDIR /home/${user_name}/data
CMD ["/bin/bash"]

