#!/usr/bin/env bash
set -e

PROJECT_ID=bk47471
./mvnw clean package -DskipTests

docker build . \
  --build-arg PARENT_IMAGE="eu.gcr.io/bolcom-stg-jvm-f30/debian-zulu-jdk-17@sha256:e0f608f103e21cf961e3977feefd7576fcc79b5eb02de4a5901aa0bf0b765432" \
  --platform linux/amd64 \
  --tag "eu.gcr.io/${PROJECT_ID}/etude-loadgen:latest"

docker push "eu.gcr.io/${PROJECT_ID}/etude-loadgen:latest"
