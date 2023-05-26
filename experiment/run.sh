#!/bin/bash
set -e

DIR="$(dirname "$0")"

DEPLOY=false
TEST=false
DESTROY=true

MACHINES=('n1-highmem-4')
ACCELERATIONS=(false) # (false 'NVIDIA_TESLA_T4')

function normalize() {
  echo "$1" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]'
}


for machine in "${MACHINES[@]}"; do
  for acceleration in "${ACCELERATIONS[@]}"; do
    model=resnet-18-custom-handler
    hardware="$(normalize "${machine}")$(if [ "${acceleration}" != "false" ]; then echo "_$(normalize "${acceleration}")"; fi)"

    if [ "${DEPLOY}" = "true" ]; then
      "${DIR}"/vertex/create_endpoint.sh "${model}_${hardware}"
      "${DIR}"/vertex/deploy_model.sh "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest" "gs://bolcom-pro-reco-analytics-fcc-shared/barrie_etude/trained/resnet-18-custom-handler/"

      if [ "${acceleration}" != "false" ]; then
        "${DIR}"/vertex/deploy_endpoint_model.sh "${model}_${hardware}" "${model}" "${machine}" "${acceleration}" '1'
      else
        "${DIR}"/vertex/deploy_endpoint_model.sh "${model}_${hardware}" "${model}" "${machine}"
      fi
    fi

    if [ "${DESTROY}" = "true" ]; then
      "${DIR}"/vertex/purge_endpoint.sh "${model}_${hardware}"
      "${DIR}"/vertex/purge_model.sh "${model}"
    fi
  done
done
