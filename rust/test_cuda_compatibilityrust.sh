#!/bin/bash

# I executed this on the 'training pod'
# python3 train.py bk47472
# ./test_cuda_compatibilitytest.sh

for d in /rust/model_store/*cuda; do
  model_filename=$(ls $d/*cuda_onnx.pth)
  payload_path=$(ls $d/*cuda_payload.yaml)

  cat << EOF > "config/serving.yaml"
  host: "0.0.0.0"
  port: 8080
  qty_actix_workers: 4
  qty_model_threads: 1
  model_path: "${model_filename}"
  payload_path: "${payload_path}"
EOF
  cat "config/serving.yaml"
  # trigger a compile so we can copy ./target/release/libonnx* in /usr/local/lib/
  cargo run --release --bin serving 2> /dev/null
  cp ./target/release/libonnx* /usr/local/lib/
  cargo run --release --bin serving config/serving.yaml &
  SERVING_PID=$!

  HOST="http://127.0.0.1:8080"
  HEALTH_URL="${HOST}/ping"

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
  kill ${SERVING_PID}
  sleep 2

done
