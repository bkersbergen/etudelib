#!/bin/bash
set -e

DIR="$(dirname "$0")"

DEPLOY=false
TEST=true
DESTROY=false

MACHINES=('n1-highmem-4')
ACCELERATIONS=(false) # (false 'NVIDIA_TESLA_T4')

function normalize() {
  echo "$1" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]'
}


for machine in "${MACHINES[@]}"; do
  for acceleration in "${ACCELERATIONS[@]}"; do
    size=1000000
    TARGET_RPS=100
    RAMP_DURATION_MINUTES=10
    model=noop_bolcom_c${size}_t50_jitopt
    hardware="$(normalize "${machine}")$(if [ "${acceleration}" != "false" ]; then echo "_$(normalize "${acceleration}")"; fi)"

    if [ "${DEPLOY}" = "true" ]; then
      "${DIR}"/vertex/create_endpoint.sh "${model}_${hardware}"
      "${DIR}"/vertex/deploy_model.sh "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest" "gs://bolcom-pro-reco-analytics-fcc-shared/barrie_etude/trained/${model}/"

      if [ "${acceleration}" != "false" ]; then
        "${DIR}"/vertex/deploy_endpoint_model.sh "${model}_${hardware}" "${model}" "${machine}" "${acceleration}" '1'
      else
        "${DIR}"/vertex/deploy_endpoint_model.sh "${model}_${hardware}" "${model}" "${machine}"
      fi
    fi

    if [ "${TEST}" = "true" ]; then
      ENDPOINT_PATH=$("$DIR"/vertex/gcloud/endpoints_state.sh | jq -r ".[] | select(.display == \"${model}_${hardware}\") | select(.models[].display == \"${model}\") | .name")
      ENDPOINT_URI="https://europe-west4-aiplatform.googleapis.com/v1/${ENDPOINT_PATH}:predict"
      REPORT_URI="gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/${model}_${hardware}.avro"
      "${DIR}"/loadgen/deploy_loadgen.sh "${ENDPOINT_URI}" "${size}" "${REPORT_URI}" "${TARGET_RPS}" "${RAMP_DURATION_MINUTES}"
    fi


    if [ "${DESTROY}" = "true" ]; then
      "${DIR}"/vertex/purge_endpoint.sh "${model}_${hardware}"
      "${DIR}"/vertex/purge_model.sh "${model}"
    fi
  done
done
