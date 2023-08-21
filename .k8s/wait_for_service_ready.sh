#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 SERVING_NAME"
    exit 1
fi

SERVING_NAME="${1}"

# Function to check if the pod is ready
check_pod_ready() {
    POD_NAME=$(kubectl get pods | grep -v Terminating | grep ${SERVING_NAME} | awk '{print $1}')
    local pod_status=$(kubectl get pod ${POD_NAME} -o jsonpath='{.status.containerStatuses[0].ready}')
    [ "$pod_status" == "true" ]
}


while ! check_pod_ready; do
  sleep 5
done
