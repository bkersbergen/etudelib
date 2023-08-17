#!/bin/bash
# set -x

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PROJECT_ID"
    exit 1
fi

DIR="$(dirname "$0")"
PROJECT_ID="${1}"

echo "$0.run(PROJECT_ID = '${PROJECT_ID}')"
PROJECT_ID="${1}"
MODEL_PATH="${2}"
PAYLOAD_PATH="${3}"


# Function to check if a file exists
file_exists() {
  if [[ $1 == gs://* ]]; then
    gsutil -q stat "$1"
  else
    [ -f "$1" ]
  fi
}

models=('core' 'gru4rec')
devices=('cpu' 'cuda')
runtimes=('jitopt' 'onnx')
c_values=(10000 1000000)
TARGET_RPS=1000
RAMP_DURATION_MINUTES=10
for DEVICE in "${devices[@]}"; do
   for RUNTIME in "${runtimes[@]}"; do
     for MODEL in "${models[@]}"; do
       for c in "${c_values[@]}"; do
         MODEL_PATH="gs://${PROJECT_ID}-shared/model_store/${MODEL}_bolcom_c${c}_t50_${DEVICE}/${MODEL}_bolcom_c${c}_t50_${DEVICE}_${RUNTIME}.pth"
         PAYLOAD_PATH="gs://${PROJECT_ID}-shared/model_store/${MODEL}_bolcom_c${c}_t50_${DEVICE}/${MODEL}_bolcom_c${c}_t50_${DEVICE}_payload.yaml"
         if ! (file_exists "$MODEL_PATH" && file_exists "$PAYLOAD_PATH"); then
           echo "Error: MODEL, payload or both do not exist."
           echo "$MODEL_PATH"
           echo "$PAYLOAD_PATH"
           exit 1
         fi
         if [ "${DEVICE}" == 'cuda' ]; then
           ${DIR}/deploy_serving_gpu.sh ${PROJECT_ID} ${MODEL_PATH} ${PAYLOAD_PATH}
         else
           ${DIR}/deploy_serving_cpu.sh ${PROJECT_ID} ${MODEL_PATH} ${PAYLOAD_PATH}
         fi
         POD_NAME=$(kubectl get pods | grep -v Terminating | grep etudelibrust | awk '{print $1}')
         echo "Waiting for pod $POD_NAME to become ready..."
         ${DIR}/wait_for_service_ready.sh ${POD_NAME}
         echo "Pod $POD_NAME is now ready."
         endpoint_ip=$(kubectl get service etudelibrust -o yaml | awk '/clusterIP:/ { gsub("\"","",$2); print $2 }')
         ${DIR}/deploy_loadgen.sh ${PROJECT_ID} "http://${endpoint_ip}:8080/predictions/model/1.0/" ${c} gs://${PROJECT_ID}-shared/results/${MODEL}_bolcom_c${c}_t50_${DEVICE}_${RUNTIME}.avro ${TARGET_RPS} ${RAMP_DURATION_MINUTES}
       done
     done
   done
done

