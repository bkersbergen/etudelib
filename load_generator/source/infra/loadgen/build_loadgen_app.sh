./mvnw package -DskipTests

docker build . \
  --build-arg PARENT_IMAGE="eu.gcr.io/bolcom-stg-jvm-f30/debian-zulu-jdk-17:latest" \
  --platform linux/amd64 \
  --tag "eu.gcr.io/bolcom-pro-reco-analytics-fcc/etude-loadgen"

docker push "eu.gcr.io/bolcom-pro-reco-analytics-fcc/etude-loadgen"
