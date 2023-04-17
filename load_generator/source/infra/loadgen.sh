#!/usr/bin/env bash
set -e

HARDWARES=('cpu' 'gpu')
RUNTIMES=('eager' 'jitopt' 'onnx')
MODELS=('core' 'gcsan' 'gru4rec' 'lightsans' 'narm' 'noop' 'repeatnet' 'sasrec' 'sine' 'srgnn' 'stamp')
CATALOG_SIZES=(1000 10000 100000 500000 1000000 5000000 10000000 20000000)

for hardware in "${HARDWARES[@]}"; do
  for runtime in "${RUNTIMES[@]}"; do
    for model in "${MODELS[@]}"; do
      for size in "${CATALOG_SIZES[@]}"; do
        echo "${hardware}-${runtime}-${model}-${size}"
      done
    done
  done
done
