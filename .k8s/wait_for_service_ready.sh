#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 POD_NAME"
    exit 1
fi

POD_NAME="${1}"

# Function to check if the pod is ready
check_pod_ready() {
    local pod_status=$(kubectl get pod ${POD_NAME} -o jsonpath='{.status.containerStatuses[0].ready}')
    [ "$pod_status" == "true" ]
}


while ! check_pod_ready; do
  sleep 5
done
