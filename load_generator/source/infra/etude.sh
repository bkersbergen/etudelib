#!/usr/bin/env bash

# setup resources
./vertex/create_endpoint.sh etude-noop
./vertex/deploy_model.sh etude-noop
./vertex/deploy_endpoint_model.sh etude-noop etude-noop

# run loadtest
# cleanup resources
