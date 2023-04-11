./mvnw package -DskipTests
docker build . --build-arg PARENT_IMAGE=eu.gcr.io/bolcom-stg-jvm-f30/debian-zulu-jdk-17:latest -t etude-loadgen

