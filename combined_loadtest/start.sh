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

echo gsutil cp "$1" /home/model-server/model-store
gsutil cp "$1" /home/model-server/model-store

cat << EOF > "/home/model-server/config.properties"
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
default_response_timeout=10
model_store=/home/model-server/model-store
workflow_store=/home/model-server/wf-store
disable_system_metrics=true
enable_metrics_api=false
vmargs=-Dlog4j.configurationFile=/home/model-server/log4j2.xml
async_logging=true
models={\
  "model": {\
    "1.0": {\
        "marName": "model.mar",\
        "batchSize": 1,\
        "maxBatchDelay": 1,\
        "responseTimeout": 100\
    }\
  }\
}
EOF

trap 'echo "torchserve.shutdown()"; exit' HUP INT QUIT TERM
torchserve --ts-config /home/model-server/config.properties --model-store /home/model-server/model-store --models all --no-config-snapshots --foreground &

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