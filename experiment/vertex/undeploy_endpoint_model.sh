#!/usr/bin/env bash


if [ $# -lt 2 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME, VERTEX_MODEL_NAME'"
    exit 1
fi

VERTEX_ENDPOINT_NAME="${1}"
VERTEX_MODEL_NAME="${2}"
DIR="$(dirname "$0")"


echo "endpoints['${VERTEX_ENDPOINT_NAME}'].undeploy(model = '${VERTEX_MODEL_NAME}')"

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
JOB_NAME="vertex-undeploy-endpoint-model-${HASH}-$(date +%s)"

ENDPOINTS_STATE=$("$DIR"/gcloud/endpoints_state.sh)

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "$VERTEX_ENDPOINT_NAME" ]; then
      ENDPOINT_EXISTS=true
      break
    fi
done

if [ "true" != "${ENDPOINT_EXISTS}" ]; then
   echo "endpoints['${VERTEX_ENDPOINT_NAME}'].404"
   exit 0
fi

MODELS_STATE=$("$DIR"/gcloud/models_state.sh)
MODEL_EXISTS=false

for MODEL in $(echo "$MODELS_STATE" | jq -r '.[].display'); do
    if [ "$MODEL" = "$VERTEX_MODEL_NAME" ]; then
      MODEL_EXISTS=true
      break
    fi
done

if [ "true" != "${MODEL_EXISTS}" ]; then
   echo "models['${VERTEX_MODEL_NAME}'].404"
   exit 0
fi

VERTEX_MODEL_DEPLOYMENT_ID=$(echo "$ENDPOINTS_STATE" | jq -r ".[] | select(.display == \"${VERTEX_ENDPOINT_NAME}\") | select(.models[].display == \"${VERTEX_MODEL_NAME}\") | .models[0].id")

if [[ "" ==  "$VERTEX_MODEL_DEPLOYMENT_ID" ]]; then
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deploy(model = '${VERTEX_MODEL_NAME}').404"
  exit
fi

echo "endpoints['${VERTEX_ENDPOINT_NAME}'].undeploy(deployment_id = '${VERTEX_MODEL_DEPLOYMENT_ID}')"
kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=60s

export VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME VERTEX_MODEL_DEPLOYMENT_ID
envsubst < "$DIR"/undeploy_endpoint_model_job.yaml > "/tmp/undeploy_endpoint_model_job.${VERTEX_ENDPOINT_NAME}_${VERTEX_MODEL_NAME}.yaml"
export -n VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME VERTEX_MODEL_DEPLOYMENT_ID

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/undeploy_endpoint_model_job.${VERTEX_ENDPOINT_NAME}_${VERTEX_MODEL_NAME}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
# Ignore any errors that may occur during the execution of the following kubectl (by appending `|| true`).
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m )

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow )
if [[ "$LOGS" =~ .*"Endpoint model undeployed.".* ]]; then
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].undeploy(model = '${VERTEX_MODEL_NAME}').ok"
  exit 0
fi

echo "$LOGS"
echo "endpoints['${VERTEX_ENDPOINT_NAME}'].undeploy(model = '${VERTEX_MODEL_NAME}').err"
exit 1
