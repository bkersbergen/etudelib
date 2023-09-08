#!/bin/bash
# set -x

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PROJECT_ID"
    exit 1
fi

# Determine the directory of the current script
export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ID="${1}"

echo "$0.run(PROJECT_ID = '${PROJECT_ID}')"

deploy_evaluate() {
  local PROJECT_ID="$1"
  local MODEL_PATH="$2"
  local PAYLOAD_PATH="$3"
  local DEVICE="$4"
  local c="$5"
  local REPORT_LOCATION="$6"
  local TARGET_RPS="$7"
  local RAMP_DURATION_MINUTES="$8"
  local sleep_delay="$9"
  if (file_exists ${REPORT_LOCATION}); then
    echo "${REPORT_LOCATION} already exists, skipping this test"
    return 0
  fi
  if ! (file_exists "$MODEL_PATH" && file_exists "$PAYLOAD_PATH"); then
    echo "Error: MODEL_PATH $MODEL_PATH, PAYLOAD_PATH $PAYLOAD_PATH or both do not exist."
    exit 1
  fi

  echo ${REPORT_LOCATION} sleeping ${sleep_delay} seconds to ramp up deployments
  sleep ${sleep_delay}

  HASH=$(echo -n "${MODEL_PATH}" | shasum | awk '{print $1}'| tr -cd '[:alnum:]')
  SECONDS=$(date +%s)
  sanitized_basename=$(basename "${REPORT_LOCATION}" | tr -cd '[:alnum:]')
  SERVING_NAME="etudeserving-${sanitized_basename}-${HASH}-${SECONDS}"
  SERVING_NAME=$(echo "${SERVING_NAME}" | tr -cd '[:alnum:]' | cut -c 1-42)
  if [ "${DEVICE}" == 'cuda' ]; then
    ${DIR}/deploy_serving_gpu.sh ${PROJECT_ID} ${MODEL_PATH} ${PAYLOAD_PATH} ${SERVING_NAME}
  else
    ${DIR}/deploy_serving_cpu.sh ${PROJECT_ID} ${MODEL_PATH} ${PAYLOAD_PATH} ${SERVING_NAME}
  fi
  ${DIR}/wait_for_service_ready.sh ${SERVING_NAME}
  POD_NAME=$(kubectl get pods | grep -v Terminating | grep ${SERVING_NAME} | awk '{print $1}')
  echo "Pod ${POD_NAME} is now ready."
  echo "Waiting for pod ${POD_NAME} to become ready..."
  endpoint_ip=$(kubectl get service ${SERVING_NAME} -o yaml | awk '/clusterIP:/ { gsub("\"","",$2); print $2 }')
  ${DIR}/deploy_loadgen.sh ${PROJECT_ID} "http://${endpoint_ip}:8080/predictions/model/1.0/" ${c} ${REPORT_LOCATION} ${TARGET_RPS} ${RAMP_DURATION_MINUTES}
  if [ $? -eq 0 ]; then
    echo "ok"
  else
    max_attempts=500  # Number of attempts (1 attempt per second)
    echo "deploy_loadgen not ok, deleting incomplete output ${REPORT_LOCATION} for max ${max_attempts} seconds"
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
      if (file_exists ${REPORT_LOCATION}); then
        gsutil -m rm ${REPORT_LOCATION}
        break
      fi
      sleep 1
      attempt=$((attempt + 1))
    done
  fi
  kubectl delete deployment ${SERVING_NAME}
  kubectl delete service ${SERVING_NAME}
}

# Function to check if a file exists
file_exists() {
  if [[ $1 == gs://* ]]; then
    gsutil -q stat "$1"
  else
    [ -f "$1" ]
  fi
}

export -f file_exists
export -f deploy_evaluate

#models=('core' 'gcsan' 'gru4rec' 'lightsans' 'narm' 'noop' 'repeatnet' 'sasrec' 'sine' 'srgnn' 'stamp' 'topkonly')
models=('core')
devices=('cuda')
#runtimes=('jitopt' 'onnx')
runtimes=('jitopt')
c_values=(10000000)
TARGET_RPS=1000
RAMP_DURATION_MINUTES=10

# Number of parallel executions
max_parallel=1
QTY_EXPERIMENT_REPEATS=1

# Initial sleep delay (seconds) for the first deployments
sleep_delay=60*${max_parallel}

for DEVICE in "${devices[@]}"; do
   for RUNTIME in "${runtimes[@]}"; do
     for MODEL in "${models[@]}"; do
       for c in "${c_values[@]}"; do
         for ((repeat=0; repeat<QTY_EXPERIMENT_REPEATS; repeat++)); do
           MODEL_PATH="gs://${PROJECT_ID}-shared/model_store/${MODEL}_bolcom_c${c}_t50_${DEVICE}/${MODEL}_bolcom_c${c}_t50_${DEVICE}_${RUNTIME}.pth"
           PAYLOAD_PATH="gs://${PROJECT_ID}-shared/model_store/${MODEL}_bolcom_c${c}_t50_${DEVICE}/${MODEL}_bolcom_c${c}_t50_${DEVICE}_payload.yaml"
           REPORT_LOCATION="gs://${PROJECT_ID}-shared/repeat_${repeat}/${MODEL}_bolcom_c${c}_t50_${DEVICE}_${RUNTIME}_rps${TARGET_RPS}.avro"
           if (file_exists ${REPORT_LOCATION}); then
             continue
           fi
           # reduce the sleep delay with each deployment
           sleep_delay=$((sleep_delay > 0 ? sleep_delay-60 : 0))
           echo "${PROJECT_ID} ${MODEL_PATH} ${PAYLOAD_PATH} ${DEVICE} ${c} ${REPORT_LOCATION} ${TARGET_RPS} ${RAMP_DURATION_MINUTES} ${sleep_delay}"
         done
       done
     done
   done
done | xargs -n 9 -P $max_parallel bash -c 'deploy_evaluate "$@"' _

