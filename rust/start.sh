#!/bin/bash
# hard failure if any required env var is not set
set -o nounset

if [[ $# -ne 2 ]]; then
    echo "Illegal number of parameters"
    echo "Usage: " $0 gs://my-storage-bucket/somedir/somemodel/model.pth s://my-storage-bucket/somedir/somemodel/config.yaml
    exit 2
fi
echo your container args are: "$@"
gcloud auth list

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

model_path_arg=$1
payload_path_arg=$2

gsutil cp "${model_path_arg}" ${DIR}/model_store/
gsutil cp "${payload_path_arg}" ${DIR}/model_store/


model_filename=$(basename -- ${model_path_arg})
payload_filename=$(basename -- ${payload_path_arg})

cat << EOF > "config/serving.yaml"
host: "0.0.0.0"
port: 8080
qty_actix_workers: 4
qty_model_threads: 1
model_path: "model_store/${model_filename}"
payload_path: "model_store/${payload_filename}"
EOF

# cargo run --release --bin serving -- model_store/noop_bolcom_c10000_t50_jitopt.pth model_store/noop_bolcom_c10000_t50_payload.yaml &
cat config/serving.yaml
# trigger a compile so we can copy ./target/release/libonnx* in /usr/local/lib/
cargo run --release --bin serving 2> /dev/null
cp ./target/release/libonnx* /usr/local/lib/
cargo run --release --bin serving config/serving.yaml &
SERVING_PID=$!

declare -r HOST="http://127.0.0.1:8080"
declare -r HEALTH_URL="${HOST}/ping"

echo "Waiting for EtudeServing to start"
sleep 1

while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' ${HEALTH_URL})" != "200" ]]; do
  echo -n .
  sleep 1;
done
echo "EtudeServing ready"

script_path="./predict_by_id.sh"
echo Doing a single prediction
# Check if the script file exists
if [ -f "$script_path" ]; then
    # Execute the script
    source "$script_path"
else
    echo "Script file '$script_path' not found."
fi

sleep infinity


