#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "requires args 'VERTEX_ENDPOINT_NAME', 'VERTEX_MODEL_NAME'"
    exit 1
fi

VERTEX_ENDPOINT_NAME="${1}"
VERTEX_MODEL_NAME="${2}"
DIR="$(dirname "$0")"

echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deploy(model = '${VERTEX_MODEL_NAME}')"

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
JOB_NAME="vertex-deploy-endpoint-model-${HASH}-$(date +%s)"

ENDPOINTS_STATE=$("$DIR"/gcloud/endpoints_state.sh)
ENDPOINT_EXISTS=false

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "${VERTEX_ENDPOINT_NAME}" ]; then
        ENDPOINT_EXISTS=true
        break
    fi
done

[ "true" != "${ENDPOINT_EXISTS}" ] && {
   echo "endpoints['${VERTEX_ENDPOINT_NAME}'].404"
   exit 1
}

MODELS_STATE=$("$DIR"/gcloud/models_state.sh)
MODEL_EXISTS=false

for MODEL in $(echo "$MODELS_STATE" | jq -r '.[].display'); do
    if [ "$MODEL" = "${VERTEX_MODEL_NAME}" ]; then
      MODEL_EXISTS=true
      break
    fi
done

[ "true" != "${MODEL_EXISTS}" ] && {
   echo "models['${VERTEX_MODEL_NAME}'].404"
   exit 1
}

MODEL_DEPLOYMENTS=$(echo "$ENDPOINTS_STATE" | jq -c "[.[] | select(.models[].display == \"${VERTEX_ENDPOINT_NAME}\")]")

[[ "0" != $(echo "${MODEL_DEPLOYMENTS}" | jq 'length') ]] && {
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deployment(model = '${VERTEX_MODEL_NAME}').200"
  exit 0
}

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

export VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME
envsubst < "$DIR"/deploy_endpoint_model_job.yaml > "/tmp/vertex-deploy-endpoint-model-${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}.yaml"
export -n VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/vertex-deploy-endpoint-model-${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=30m)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
[[ "$LOGS" =~ .*"deployed to Endpoint".* ]] && {
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deploy(model = '${VERTEX_MODEL_NAME}').ok"
  exit 0
}

echo "$LOGS"
echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deploy(model = '${VERTEX_MODEL_NAME}').err"
exit 1
