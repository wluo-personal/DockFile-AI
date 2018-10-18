ARG ubuntu_version=18.04
FROM ubuntu:${ubuntu_version}
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
    && echo "export PATH=/home/${user_name}/anaconda3/bin:$PATH" >> ~/.bashrc
    
# install important packages
USER ${user_name}
RUN apt-get update \
    && git config --global user.name "kai" \
    && git config --global user.email "kai@katabat.com"

# Create python CPU version
# In bashrc, it will return at the beginning unless it is running at interactive mode! source cmd will not work in docker build!
ARG python_cpu_version='3.6'
USER ${user_name}
# link bash with /bin/sh. RUN command default to use /bin/sh instead of /bin/bash
RUN ["/bin/bash", "-c", "echo $password_root | sudo -S ln -sf bash /bin/sh"]
RUN export PATH=/home/${user_name}/anaconda3/bin:$PATH \
    && yes | conda create -n python${python_cpu_version}-CPU python=${python_cpu_version} \
    && source activate python${python_cpu_version}-CPU \
    && pip install --upgrade pip \
    && yes | pip install lightgbm \
    && yes | pip install xgboost \
    && yes | pip install catboost \
    && yes | pip install --upgrade tensorflow \
    && yes | pip install keras \
    && source deactivate


# link bash with /bin/sh. RUN command default to use /bin/sh instead of /bin/bash
# RUN ["/bin/bash", "-c", "echo $password_root | sudo -S ln -sf bash /bin/sh"]
# revoke bash with /bin/sh
RUN ["/bin/bash", "-c", "echo $password_root | sudo -S ln -sf dash /bin/sh"]

# run with sudo priviledge
# echo $password_root | sudo -S <cmd>
#echo $password_root | sudo -S -H -u <user> <cmd>
# link bash with /bin/sh 
# sudo ln -sf bash /bin/sh
# revert system to default
# sudo ln -sf dash /bin/sh
CMD ["/bin/bash"]

