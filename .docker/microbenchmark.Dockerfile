# FROM eu.gcr.io/bk47472/etudelib/serving_rust:latest
ARG PARENT_IMAGE
FROM $PARENT_IMAGE

RUN pip3 install onnx==1.14.0 model_archiver
USER root
WORKDIR /

COPY etudelib/ ./etudelib/
COPY setup.py ./
COPY pyproject.toml ./
COPY README.md ./
COPY requirements/ ./requirements/
RUN pip3 install -e .
ENV PYTHONPATH=./etudelib:$PYTHONPATH

RUN mkdir /.config
# free up disk space
RUN apt-get autoclean


CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
