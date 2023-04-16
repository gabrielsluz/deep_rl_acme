FROM python:3.9-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install -y swig && \
    apt install -y htop && \
    apt install -y ffmpeg libsm6 libxext6

COPY ./req.txt /req.txt
RUN pip install --no-cache-dir -r /req.txt

CMD ['/bin/bash']
EXPOSE 8080