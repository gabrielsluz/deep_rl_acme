#! /bin/bash

# From: https://github.com/eustin/docker-tf-gcp/blob/master/startup.sh

STARTUP_SUCCESS_FILE=/home/.ran-startup-script

if test ! -f "$STARTUP_SUCCESS_FILE"; then
	echo "$STARTUP_SUCCESS_FILE does not exist. running startup..."

	# host machine requires nvidia drivers. tensorflow image should contain the rest required
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
	sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
	sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
	sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
	sudo apt-get update && sudo apt-get install -y cuda-drivers

	# install docker
	sudo apt-get update && sudo apt-get install -y \
	    apt-transport-https \
	    ca-certificates \
	    curl \
	    gnupg-agent \
	    software-properties-common

	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
	sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io

	# install nvidia docker support
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
	curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
	curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
	sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
	sudo systemctl restart docker

	# create file which will be checked on next reboot
	touch /home/.ran-startup-script
else
	echo "$STARTUP_SUCCESS_FILE exists. not running startup script!"
fi