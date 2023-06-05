#!/bin/bash
# hard failure if any required env var is not set
set -o nounset

if [[ $# -ne 1 ]]; then
    echo "Illegal number of parameters"
    echo "Usage: " $0 gs://my-storage-bucket/model_store/noop_bolcom_c1000000_t50_jitopt/model.mar
    exit 2
fi
echo your container args are: "$@"

echo gsutil cp "$1" /app/models
gsutil cp "$1" /app/models

trap 'echo "torchserve.shutdown()"; exit' HUP INT QUIT TERM
torchserve --ts-config /app/conf/torchserve.properties --model-store /app/models --models all --no-config-snapshots --foreground &

url="http://127.0.0.1:7080/ping"
status_code=""
echo "Waiting for torchserve to be started"

while [[ "$status_code" != "200" ]]; do
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    status_code=$(echo "$response" | awk '{print $1}')
    if [[ "$status_code" != "200" ]]; then
        echo "Received HTTP status code $status_code. Retrying..."
        sleep 1  # Wait for 1 second before retrying
    fi
done

echo "Received HTTP status code $status_code for $url. Continuing..."



sleep infinity &
wait $!
