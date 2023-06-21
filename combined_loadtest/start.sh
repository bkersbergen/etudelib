#!/bin/bash
# hard failure if any required env var is not set
set -o nounset

if [[ $# -ne 1 ]]; then
    echo "Illegal number of parameters"
    echo "Usage: " $0 gs://my-storage-bucket/model_store/noop_bolcom_c1000000_t50_jitopt/model.mar
    exit 2
fi
echo your container args are: "$@"
gcloud auth list

echo gsutil cp "$1" /home/pytorch/models
gsutil cp "$1" /home/pytorch/models

cat << EOF > "/home/pytorch/config.properties"
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
async_logging=true
vmargs=-Dlog4j.configurationFile=/home/pytorch/conf/log4j2.xml
default_response_timeout=1000
enable_metrics_api=false
load_models=all
model_store=/home/pytorch/models
enable_envvars_config=true
models={"model": {"1.0": {"batchSize": 1,"marName": "model.mar","maxBatchDelay": 1,"responseTimeout": 100}}}
EOF

trap 'echo "torchserve.shutdown()"; exit' HUP INT QUIT TERM
torchserve --ts-config /home/pytorch/config.properties --model-store /home/pytorch/models --models all --no-config-snapshots --foreground &

url="http://127.0.0.1:8080/ping"
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


# curl -X POST -H "Content-Type: application/json" http://localhost:8080/predictions/model/1.0/ --data """{"instances": [{"context": [1,2,3]}],"parameters": [{"runtime":  ""}]}"""