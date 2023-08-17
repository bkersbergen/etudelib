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
devices=('cpu' 'gpu')
runtimes=('jitopt' 'onnx')
c_values=(10000 1000000)

for device in "${devices[@]}"; do
   for runtime in "${runtimes[@]}"; do
     for model in "${models[@]}"; do
       for c in "${c_values[@]}"; do
         MODEL_PATH="gs://${PROJECT_ID}-shared/model_store/${model}_bolcom_c${c}_t50_${device}/${model}_bolcom_c${c}_t50_${device}_${runtime}.pth"
         PAYLOAD_PATH="gs://${PROJECT_ID}-shared/model_store/${model}_bolcom_c${c}_t50_${device}/${model}_bolcom_c${c}_t50_${device}_payload.yaml"
         # Check if both files exist
         if ! (file_exists "$MODEL_PATH" && file_exists "$PAYLOAD_PATH"); then
           echo "Error: One or both files do not exist."
           echo "$MODEL_PATH"
           echo "$PAYLOAD_PATH"
           exit 1
         fi
         echo "deploy model $MODEL_PATH $PAYLOAD_PATH"
       done
     done
   done
done