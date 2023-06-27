# Running in GCloud
https://embracingtherandom.com/deep-learning/cloud/tensorflow/docker/dockerise-your-tf/#do-ya-got-some-gpu-quota

```
gcloud compute instances create t4-instance-1 \
    --project=northern-bot-383915 \
    --zone=us-west1-b \
    --machine-type=n1-standard-4 \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=767806385608-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-t4 \
    --create-disk=auto-delete=yes,boot=yes,device-name=t4-instance-1,image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20230411-debian-10-py37,mode=rw,size=50,type=projects/northern-bot-383915/zones/us-west1-b/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=ec-src=vm_add-gcloud \
    --reservation-affinity=any \
	--metadata-from-file startup-script=./startup.sh
```

## Installing in the VM:
Create VM.
Answer y for install NVIDIA drivers.



```
pip install dm-acme[jax,tf,envs]==0.4.0
pip install box2d-py==2.3.8
pip install opencv-python==4.5.5.62
```

Test GPU:
```
python -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"
```

Test GPU jax:


Issues:
- Error connecting SSH firewall:
https://stackoverflow.com/questions/63147497/connection-via-cloud-identity-aware-proxy-failed

Drivers:
DRIVER_VERSION: 510.47.03
Downloading driver from GCS location and install: gs://nvidia-drivers-us-public/tesla/510.47.03/NVIDIA-Linux-x86_64-510.47.03.run

>>> import jax
>>> jax.devices()
WARNING:jax._src.lib.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
[CpuDevice(id=0)]
>>> jax.__version__
'0.3.25'

conda create --name acme_env python=3.9 
conda activate acme_env

sudo apt-get install swig
sudo apt-get install -y ffmpeg libsm6 libxext6

git clone https://github.com/gabrielsluz/deep_rl_acme.git
pip install -r req.txt
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Jax with GPU support in GCP:
Jax instructions:

You must first install the NVIDIA driver. We recommend installing the newest driver available from NVIDIA, but the driver must be version >= 525.60.13 for CUDA 12 and >= 450.80.02 for CUDA 11 on Linux.

JAX currently ships three CUDA wheel variants:

CUDA 12.0 and CuDNN 8.8.
CUDA 11.8 and CuDNN 8.6.
CUDA 11.4 and CuDNN 8.2. This wheel is deprecated and will be discontinued with jax 0.4.8.

What do we need to run Jax with GPU?
- Cuda drivers
- cuDNN: Cuda library for Deep Learning
- CUDA Toolkit => everything we need?

What jax version do I need?
jax==0.4.1
jaxlib==0.4.1
=> works with ACME

Ideia Container:
- Create Ubuntu VM with Docker.
- Install GPU drivers
- Run a container with Jax and ACME.

Base image: nvidia/cuda:11.8.0-devel-ubuntu20.04

## Instalation:
```
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# install python3-pip
RUN apt update && apt install python3-pip -y

RUN pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
pip install dm-acme[jax,tf,envs]==0.4.0
pip install distrax==0.1.2
pip install scikit-image==0.19.3

sudo apt-get install -y swig
sudo apt-get install -y ffmpeg libsm6 libxext6

pip install box2d-py==2.3.8
pip install opencv-python==4.5.5.62