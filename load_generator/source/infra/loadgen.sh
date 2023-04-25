#!/usr/bin/env bash
set -e

DIR="$(dirname "$0")"

DEPLOY=false
TEST=true
DESTROY=false

MACHINES=('n1-highmem-4')
ACCELERATIONS=(false) # (false 'NVIDIA_TESLA_T4')
RUNTIMES=('eager') # ('eager' 'jitopt' 'onnx')
NETWORKS=('noop') # ('noop' 'random' 'core' 'gcsan' 'gru4rec' 'lightsans' 'narm' 'repeatnet' 'sasrec' 'sine' 'srgnn' 'stamp')
CATALOG_SIZES=(1000000) # (1000 10000 100000 500000 1000000 5000000)

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
              ENDPOINT_PATH=$(./vertex/gcloud/endpoints_state.sh | jq -r ".[] | select(.display == \"${model}_${hardware}\") | select(.models[].display == \"${model}\") | .models[0].name")
              ENDPOINT_URI="https://europe-west4-aiplatform.googleapis.com/v1/${ENDPOINT_PATH}:predict"
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

# noop 1k & 5m
# ./infra/loadgen/deploy_loadgen.sh "https://europe-west4-aiplatform.googleapis.com/v1/projects/1077776595046/locations/europe-west4/endpoints/7748610284658360320:predict" "1000" "gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/noop_bolcom_c1000_t50_eager_n1highmem4.avro"
# ./infra/loadgen/deploy_loadgen.sh "https://europe-west4-aiplatform.googleapis.com/v1/projects/1077776595046/locations/europe-west4/endpoints/1677757986962931712:predict" "5000000" "gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/noop_bolcom_c5000000_t50_eager_n1highmem4.avro"

# random 1k & 5m
# ./infra/loadgen/deploy_loadgen.sh "https://europe-west4-aiplatform.googleapis.com/v1/projects/1077776595046/locations/europe-west4/endpoints/2515427517653843968:predict" "1000" "gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/random_bolcom_c1000_t50_eager_n1highmem4.avro"
# ./infra/loadgen/deploy_loadgen.sh "https://europe-west4-aiplatform.googleapis.com/v1/projects/1077776595046/locations/europe-west4/endpoints/5063339006838702080:predict" "5000000" "gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/random_bolcom_c5000000_t50_eager_n1highmem4.avro"
