#!/usr/bin/env bash
set -e

DIR="$(dirname "$0")"

DEPLOY=true
TEST=false
DESTROY=false

MACHINES=('n1-highmem-4')
ACCELERATIONS=(false 'NVIDIA_TESLA_T4')
RUNTIMES=('eager' 'jitopt' 'onnx')
NETWORKS=('noop' 'random') # ('noop' 'random' 'core' 'gcsan' 'gru4rec' 'lightsans' 'narm' 'repeatnet' 'sasrec' 'sine' 'srgnn' 'stamp')
CATALOG_SIZES=(1000 5000000) # (1000 10000 100000 500000 1000000 5000000)

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
            "${DIR}"/vertex/create_endpoint.sh "${model}_${hardware}"
            "${DIR}"/vertex/deploy_model.sh "${model}" "eu.gcr.io/bolcom-pro-reco-analytics-fcc/etudelib/${model}:latest"

            if [ "${acceleration}" != "false" ]; then
              "${DIR}"/vertex/deploy_endpoint_model.sh "${model}_${hardware}" "${model}" "${machine}" "${acceleration}" '1'
            else
              "${DIR}"/vertex/deploy_endpoint_model.sh "${model}_${hardware}" "${model}" "${machine}"
            fi
          fi

          if [ "${TEST}" = "true" ]; then
              ENDPOINT_URI="https://europe-west4-aiplatform.googleapis.com/v1/$(./vertex/gcloud/endpoints_state.sh | jq -r ".[] | select(.display == \"${model}_${hardware}\") and select(.models[].display == \"${model}\") | .name"):predict"
              REPORT_URI="gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/${model}_${hardware}.avro"
              "${DIR}"/loadgen/deploy_loadgen.sh "${ENDPOINT_URI}" "${size}" "${REPORT_URI}"
          fi

          if [ "${DESTROY}" = "true" ]; then
            "${DIR}"/vertex/purge_endpoint.sh "${model}_${hardware}"
            "${DIR}"/vertex/purge_model.sh "${model}"
          fi
        done
      done
    done
  done
done

#./loadgen/deploy_loadgen.sh "https://europe-west4-aiplatform.googleapis.com/v1/projects/1077776595046/locations/europe-west4/endpoints/7748610284658360320:predict" "1000" "gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/noop_bolcom_c1000_t50_eager_n1highmem4.avro"
