#!/usr/bin/env bash
set -e

HARDWARES=('cpu') # ('cpu' 'gpu')
RUNTIMES=('eager') # ('eager' 'jitopt' 'onnx')
MODELS=('core') # ('noop' 'core' 'gcsan' 'gru4rec' 'lightsans' 'narm' 'repeatnet' 'sasrec' 'sine' 'srgnn' 'stamp')
CATALOG_SIZES=(100000) # (1000 10000 100000 500000 1000000 5000000 10000000 20000000)

for hardware in "${HARDWARES[@]}"; do
  for runtime in "${RUNTIMES[@]}"; do
    for model in "${MODELS[@]}"; do
      for size in "${CATALOG_SIZES[@]}"; do
        ./vertex/create_endpoint.sh "$model"
        ./vertex/deploy_model.sh "$model" "eu.gcr.io/bolcom-pro-reco-analytics-fcc/${model}_bolcom_c${size}_t50_${runtime}:latest"
        if [ "$hardware" = "gpu" ]; then
           ./vertex/deploy_endpoint_model.sh "$model" "$model" 'n1-highmem-4' 'NVIDIA_TESLA_T4' '1'
        else
          ./vertex/deploy_endpoint_model.sh "$model" "$model" 'n1-highmem-4'
        fi
        echo "loadtest.run()"
        sleep 60
#        ./loadgen/deploy_loadgen.sh "$model-$size-$runtime-$hardware" "$runtime" "$size"
        ./vertex/purge_endpoint.sh "$model"
        ./vertex/purge_model.sh "$model"
      done
    done
  done
done
