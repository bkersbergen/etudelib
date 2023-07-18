#!/bin/bash
# hard failure if any required env var is not set
set -o nounset

DIR="$(dirname "$0")"
PROJECT_ID=bk47471
echo Determining service endpoint for etudelibrust
endpoint_ip=$(kubectl get service etudelibrust -o yaml | awk '/clusterIP:/ { gsub("\"","",$2); print $2 }')
echo Found endpoint: http://${endpoint_ip}:8080
"${DIR}"/loadgen/deploy_loadgen.sh http://${endpoint_ip}:8080/predictions/model/1.0/ 100000 gs://${PROJECT_ID}-shared/results/static.avro 1000 10
