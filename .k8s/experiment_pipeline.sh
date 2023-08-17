#!/bin/bash
# set -x

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PROJECT_ID"
    exit 1
fi

DIR="$(dirname "$0")"
PROJECT_ID="${1}"

echo "$0.run(PROJECT_ID = '${PROJECT_ID}')"

models=('core' 'gru4rec')
devices=('cpu' 'gpu')
runtimes=('jit' 'onnx')
c_values=(10000 1000000 10000000)

for device in "${devices[@]}"; do
   for runtime in "${runtimes[@]}"; do
     for model in "${models[@]}"; do
       for c in "${c_values[@]}"; do
         echo "$model $device $runtime $c"
       done
     done
   done
done