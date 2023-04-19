#!/usr/bin/env bash
set -e

DEPLOY=false
TEST=false
DESTROY=false

MACHINES=('n1-highmem-4')
ACCELERATIONS=(false 'NVIDIA_TESLA_T4')
RUNTIMES=('eager') # 'jitopt' 'onnx')
NETWORKS=('noop') # ('noop' 'core' 'gcsan' 'gru4rec' 'lightsans' 'narm' 'repeatnet' 'sasrec' 'sine' 'srgnn' 'stamp')
CATALOG_SIZES=(1000) # 10000 100000 500000 1000000 5000000)

function normalize() {
  echo "$1" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]'
}

for machine in "${MACHINES[@]}"; do
  for acceleration in "${ACCELERATIONS[@]}"; do
    for runtime in "${RUNTIMES[@]}"; do
      for network in "${NETWORKS[@]}"; do
        for size in "${CATALOG_SIZES[@]}"; do
          model="${network}_bolcom_c${size}_t50_${runtime}"
          hardware="$(normalize "${machine}")$(if [ "${acceleration}" != "false" ]; then echo "_$(normalize "${acceleration}")"; fi)"

          if [ "${DEPLOY}" = "true" ]; then
            ./vertex/create_endpoint.sh "${model}_${hardware}"
            ./vertex/deploy_model.sh "${model}" "eu.gcr.io/bolcom-pro-reco-analytics-fcc/${model}:latest"

            if [ "${acceleration}" != "false" ]; then
              ./vertex/deploy_endpoint_model.sh "${model}_${hardware}" "${model}" "${machine}" "${acceleration}" '1'
            else
              ./vertex/deploy_endpoint_model.sh "${model}_${hardware}" "${model}" "${machine}"
            fi
          fi

          if [ "${TEST}" = "true" ]; then
              ENDPOINT_URI="https://europe-west4-aiplatform.googleapis.com/v1/$(./vertex/gcloud/endpoints_state.sh | jq -r ".[] | select(.display == \"${model}_${hardware}\").model" ):predict"
              REPORT_URI="gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/${model}_${hardware}.avro"
              ./loadgen/deploy_loadgen.sh "${ENDPOINT_URI}" "${size}" "${REPORT_URI}"
          fi

          if [ "${DESTROY}" = "true" ]; then
            ./vertex/purge_endpoint.sh "${model}_${hardware}"
            ./vertex/purge_model.sh "${model}"
          fi
        done
      done
    done
  done
done
