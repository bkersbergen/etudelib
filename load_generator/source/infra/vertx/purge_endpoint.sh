#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTX_ENDPOINT_NAME_OR_ID'"
    exit 1
fi

export VERTX_ENDPOINT_NAME_OR_ID="${1}"
ENDPOINTS_STATE=$(./gcloud/endpoints_state.sh)

for name in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$name" = "$VERTX_ENDPOINT_NAME_OR_ID" ]; then
      ENDPOINT_EXISTS=true
      break
    fi
done

 [ "true" != "${ENDPOINT_EXISTS}" ] && {
   echo "endpoint(name = '${VERTX_ENDPOINT_NAME_OR_ID}').404"
   exit 1
}

ENDPOINT_MODELS=$(echo "$ENDPOINTS_STATE" | jq "[.[] | select(.display == \"${VERTX_ENDPOINT_NAME_OR_ID}\").models[]]")
echo "$ENDPOINT_MODELS" | jq .

for model_id in $(echo "$ENDPOINT_MODELS" | jq -r '.[].id'); do
 echo "unlink model(id = ${model_id}) from endpoint(name = ${VERTX_ENDPOINT_NAME_OR_ID})"
 echo $(./unlink_endpoint_model.sh "${VERTX_ENDPOINT_NAME_OR_ID}" "${model_id}")
done




#echo $(echo $ENDPOINTS_STATE | jq -r '.[].display')
#echo $ENDPOINT_EXISTS
#exit 0;





#
#
#HASH=$(sum <<< "${VERTX_ENDPOINT_NAME_OR_ID}" | cut -f 1 -d ' ')
#export JOB_NAME="etude-vertex-delete-endpoint-${HASH}"
#
#kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=10m
#
#envsubst < ./delete_endpoint_job.yaml > "/tmp/delete_endpoint_job.${VERTX_ENDPOINT_NAME_OR_ID}.yaml"
#cat "/tmp/delete_endpoint_job.${VERTX_ENDPOINT_NAME_OR_ID}.yaml"
#
#kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/delete_endpoint_job.${VERTX_ENDPOINT_NAME_OR_ID}.yaml"
