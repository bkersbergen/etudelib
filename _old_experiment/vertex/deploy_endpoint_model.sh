#!/usr/bin/env bash

if [ $# -lt 3 ]; then
    echo "requires args 'VERTEX_ENDPOINT_NAME', 'VERTEX_MODEL_NAME', 'VERTEX_MACHINE'"
    exit 1
fi

VERTEX_ENDPOINT_NAME="${1}"
VERTEX_MODEL_NAME="${2}"
VERTEX_MACHINE="${3}"
VERTEX_ACCELERATOR="${4}"
VERTEX_ACCELERATOR_COUNT="${5}"

if [ $# -gt 3 ]; then
  if [ $# -lt 5 ]; then
    echo "requires args 'VERTEX_ENDPOINT_NAME', 'VERTEX_MODEL_NAME', 'VERTEX_MACHINE', 'VERTEX_ACCELERATOR', 'VERTEX_ACCELERATOR_COUNT'"
    exit 1
  fi
fi


DIR="$(dirname "$0")"

echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deploy(model = '${VERTEX_MODEL_NAME}')"

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}-${VERTEX_MACHINE}-${VERTEX_ACCELERATOR}-${VERTEX_ACCELERATOR_COUNT}" | cut -f 1 -d ' ')
JOB_NAME="vertex-deploy-endpoint-model-${HASH}-$(date +%s)"

ENDPOINTS_STATE=$("$DIR"/gcloud/endpoints_state.sh)
ENDPOINT_EXISTS=false

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "${VERTEX_ENDPOINT_NAME}" ]; then
        ENDPOINT_EXISTS=true
        break
    fi
done

if [ "true" != "${ENDPOINT_EXISTS}" ]; then
   echo "endpoints['${VERTEX_ENDPOINT_NAME}'].404"
   exit 1
fi

MODELS_STATE=$("$DIR"/gcloud/models_state.sh)
MODEL_EXISTS=false

for MODEL in $(echo "$MODELS_STATE" | jq -r '.[].display'); do
    if [ "$MODEL" = "${VERTEX_MODEL_NAME}" ]; then
      MODEL_EXISTS=true
      break
    fi
done

if [ "true" != "${MODEL_EXISTS}" ]; then
   echo "models['${VERTEX_MODEL_NAME}'].404"
   exit 1
fi

MODEL_DEPLOYMENT=$(echo "$ENDPOINTS_STATE" | jq -c "[.[] | select(.display == \"${VERTEX_ENDPOINT_NAME}\") | select(.models[].display == \"${VERTEX_MODEL_NAME}\")]")

if [[ "0" != $(echo "${MODEL_DEPLOYMENT}" | jq 'length') ]]; then
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deployment(model = '${VERTEX_MODEL_NAME}').200"
  exit 0
fi

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

export VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME VERTEX_MACHINE VERTEX_ACCELERATOR VERTEX_ACCELERATOR_COUNT
if [ $# -lt 5 ]; then
  envsubst < "$DIR"/deploy_endpoint_model_job.yaml > "/tmp/vertex-deploy-endpoint-model-${VERTEX_ENDPOINT_NAME}-${HASH}.yaml"
else
  envsubst < "$DIR"/deploy_accelerated_endpoint_model_job.yaml > "/tmp/vertex-deploy-endpoint-model-${VERTEX_ENDPOINT_NAME}-${HASH}.yaml"
fi
export -n VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME VERTEX_MACHINE VERTEX_ACCELERATOR VERTEX_ACCELERATOR_COUNT

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/vertex-deploy-endpoint-model-${VERTEX_ENDPOINT_NAME}-${HASH}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')

# Ignore any errors that may occur during the execution of the following kubectl (by appending `|| true`).
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=30m )
LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow )
if [[ "$LOGS" =~ .*"deployed to Endpoint".* ]]; then
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deploy(model = '${VERTEX_MODEL_NAME}').ok"
  exit 0
fi

echo "$LOGS"
echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deploy(model = '${VERTEX_MODEL_NAME}').err"
exit 1
