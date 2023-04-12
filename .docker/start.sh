#!/bin/bash
# hard failure if any required env var is not set
set -o nounset

trap 'echo "torchserve.shutdown()"; exit' HUP INT QUIT TERM
torchserve --ts-config /app/conf/torchserve.properties --model-store /app/models --models all --no-config-snapshots --foreground &
sleep infinity &
wait $!
