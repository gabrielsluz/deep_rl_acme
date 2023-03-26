# ACME
https://github.com/deepmind/acme

## Installation:
Requires a Linux based OS.
Docker tutorial: https://towardsdatascience.com/a-complete-guide-to-building-a-docker-image-serving-a-machine-learning-system-in-production-d8b5b0533bde

Docker:

```
docker build -t acme_im .
docker run -it -v /Users/zeba/Desktop/Mestrado/Dissertacao/box_pushing/acme_dev:/shared_dir  acme_im /bin/bash
```


Modificacoes no container:
- Installar htop
- Instalar Box2D e opencv
```
pip install box2d-py==2.3.8
pip install opencv-python==4.5.5.62
```
Colocar no Docker para dependencias do OpenCV:
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


Quero que esse contianer funcione como um ambiente para rodar diferentes programas. Ou seja, tenho que passar o código por fora.
Singularity container com um diretório compartilhado => código tem esse diretorio parametrizado e faz o que quiser lá.

Falta:
- Código para rodar um teste com ACME => treinar uma política com sucesso.
- Mecanismos na image docker para rodar experimentos e guardar os resultados
    - Tensorboard
- Integrar com singularity VM
- Rodar no Verlab => talvez precise de um admin para mudar a permissão da sandbox. Mas e se eu fizer a transferência de arquivos e mantar o container .sif direto para meu home?

<!-- Environment:
```
conda create --name acme_env python=3.9
conda activate acme_env

pip install --upgrade pip setuptools wheel
pip install git+https://github.com/deepmind/rlax.git
pip install dm-acme[jax,tf,envs]
pip install git+https://github.com/deepmind/acme.git#egg=dm-acme[jax,tf,envs]
``` -->
