#!/bin/bash
# set -x

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 PROJECT_ID MODEL_PATH PAYLOAD_PATH"
    exit 1
fi

DIR="$(dirname "$0")"
PROJECT_ID="${1}"
MODEL_PATH="${2}"
PAYLOAD_PATH="${3}"
YAML_TEMPLATE=${DIR}/etudelibrust-deployment_gpu.yaml

echo "$0.run(PROJECT_ID = '${PROJECT_ID}', MODEL_PATH = '${MODEL_PATH}', PAYLOAD_PATH = '${PAYLOAD_PATH}'')"

# Check if the YAML file exists
if [ ! -f "$YAML_TEMPLATE" ]; then
    echo "Error: YAML file $YAML_TEMPLATE not found."
    exit 1
fi

kubectl delete deployment etudelibrust --grace-period=0 --wait=true --ignore-not-found=true --timeout=5m

kubectl apply -f <(
  sed -e "s|\${PROJECT_ID}|${PROJECT_ID}|" \
    -e "s|\${MODEL_PATH}|${MODEL_PATH}|" \
    -e "s|\${PAYLOAD_PATH}|${PAYLOAD_PATH}|" \
    $YAML_TEMPLATE
    )

kubectl apply -f ${DIR}/etudelibrust-service.yaml
