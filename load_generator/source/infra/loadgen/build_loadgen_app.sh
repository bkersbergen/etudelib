#!/usr/bin/env bash
set -e

PROJECT_ID=bk47479
./mvnw clean package -DskipTests

docker build . \
  --build-arg PARENT_IMAGE="eu.gcr.io/bolcom-stg-jvm-f30/debian-zulu-jdk-17:latest" \
  --platform linux/amd64 \
  --tag "eu.gcr.io/${PROJECT_ID}/etude-loadgen:latest"

docker push "eu.gcr.io/${PROJECT_ID}/etude-loadgen:latest"
