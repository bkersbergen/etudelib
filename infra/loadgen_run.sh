#!/bin/bash
# hard failure if any required env var is not set
set -o nounset

if [[ $# -ne 1 ]]; then
    echo "Illegal number of parameters. PROJECT_ID is missing"
    echo "Usage: $0 bk47471"
    exit 2
fi

PROJECT_ID=$1

DIR="$(dirname "$0")"
echo Determining service endpoint for etudelibrust
endpoint_ip=$(kubectl get service etudelibrust -o yaml | awk '/clusterIP:/ { gsub("\"","",$2); print $2 }')
echo Found endpoint: http://${endpoint_ip}:8080
"${DIR}"/../.k8s/deploy_loadgen.sh ${PROJECT_ID} http://${endpoint_ip}:8080/predictions/model/1.0/ 10000 gs://${PROJECT_ID}-shared/results/static.avro 1000 10