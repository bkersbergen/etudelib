#!/usr/bin/env bash
set -e

CREATE=true
DESTROY=false

HARDWARES=('cpu') # ('cpu' 'gpu')
RUNTIMES=('eager') # ('eager' 'jitopt' 'onnx')
MODELS=('core') # ('noop' 'core' 'gcsan' 'gru4rec' 'lightsans' 'narm' 'repeatnet' 'sasrec' 'sine' 'srgnn' 'stamp')
CATALOG_SIZES=(100000) # (1000 10000 100000 500000 1000000 5000000 10000000 20000000)

for hardware in "${HARDWARES[@]}"; do
  for runtime in "${RUNTIMES[@]}"; do
    for model in "${MODELS[@]}"; do
      for size in "${CATALOG_SIZES[@]}"; do
        [ "${CREATE}" != "false" ] && {
          ./vertex/create_endpoint.sh "$model"
          ./vertex/deploy_model.sh "$model" "eu.gcr.io/bolcom-pro-reco-analytics-fcc/${model}_bolcom_c${size}_t50_${runtime}:latest"

          if [ "$hardware" = "gpu" ]; then
             ./vertex/deploy_endpoint_model.sh "$model" "$model" 'n1-highmem-4' 'NVIDIA_TESLA_T4' '1'
          else
            ./vertex/deploy_endpoint_model.sh "$model" "$model" 'n1-highmem-4'
          fi
        }

        ENDPOINT_URI="https://europe-west4-aiplatform.googleapis.com/v1/$(./vertex/gcloud/endpoints_state.sh | jq -r ".[] | select(.display == \"${model}\").name" ):predict"
        REPORT_URI="gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/${model}-bolcom-c${size}-t50-${runtime}.avro"
        ./loadgen/deploy_loadgen.sh "${ENDPOINT_URI}" "$size" "${REPORT_URI}"

        [ "${DESTROY}" = "true" ] && {
          ./vertex/purge_endpoint.sh "$model"
          ./vertex/purge_model.sh "$model"
        }
      done
    done
  done
done
