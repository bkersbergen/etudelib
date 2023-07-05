#!/bin/bash

declare -r HOST="http://127.0.0.1:8080"
declare -r HEALTH_URL="${HOST}/ping"
# cargo run --release --bin serving -- model_store/noop_bolcom_c10000_t50_jitopt.pth model_store/noop_bolcom_c10000_t50_payload.yaml &
cat config/serving.yaml
target/release/serving config/serving.yaml &
SERVING_PID=$!


echo -n "Waiting for http server to start "
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' ${HEALTH_URL})" != "200" ]]; do
  echo -n .
  sleep 1;
done
echo EtudeLib Model serving ready

sleep infinity


