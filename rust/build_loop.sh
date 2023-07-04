#!/bin/bash
max_attempts=5
attempt=0
while [ $attempt -lt $max_attempts ]; do
    cargo build --release --bin serving
#    cargo build --bin serving
    return_code=$?
    if [ $return_code -eq 0 ]; then
        echo "Build successful"
        break
    else
        echo "Build failed with return code $return_code. Retrying..."
        sleep 1
        ((attempt++))
    fi
done