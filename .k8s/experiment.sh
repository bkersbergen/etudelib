#!/bin/bash
# set -x

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PROJECT_ID"
    exit 1
fi

DIR="$(dirname "$0")"
PROJECT_ID="${1}"

echo "$0.run(PROJECT_ID = '${PROJECT_ID}')"

devices=('cpu' 'gpu')
runtimes=('jit' 'onnx')

for device in "${devices[@]}"; do
   for runtime in "${runtimes[@]}"; do
      echo "$device $runtime"
   done
done