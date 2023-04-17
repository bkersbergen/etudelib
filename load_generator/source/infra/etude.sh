#!/usr/bin/env bash
set -e

# setup resources
./vertex/create_endpoint.sh 'etude-noop'
./vertex/deploy_model.sh 'etude-noop' 'eu.gcr.io/bolcom-pro-reco-analytics-fcc/etude-noop:latest'

./vertex/deploy_endpoint_model.sh 'etude-noop' 'etude-noop' 'n1-highmem-4'
#./vertex/deploy_endpoint_model.sh 'etude-noop' 'etude-noop' 'n1-highmem-4' 'NVIDIA_TESLA_T4' '1'

# run loadtest
# ...

# cleanup resources
./vertex/purge_endpoint.sh 'etude-noop'
./vertex/purge_model.sh 'etude-noop'
