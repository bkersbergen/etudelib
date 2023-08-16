# FROM eu.gcr.io/bk47472/etudelib/serving_rust:latest
ARG PARENT_IMAGE
FROM $PARENT_IMAGE

RUN pip install onnx==1.14.0
USER root

COPY ./etudelib etudelib
COPY ./rust/train.py ./

RUN mkdir /.config
# free up disk space
RUN apt-get autoclean

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
