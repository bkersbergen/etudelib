#!/usr/bin/env bash


PROJECT="bolcom-pro-reco-analytics-fcc"
USER=${USER:-SA}

IMAGE_URI=eu.gcr.io/${PROJECT}/etudelib-microbenchmark:main

docker build -t ${IMAGE_URI} -f .ci/Dockerfile .
docker push ${IMAGE_URI}

