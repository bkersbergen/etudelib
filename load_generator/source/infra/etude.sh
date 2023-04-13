#!/usr/bin/env bash
set -e

# setup resources
./vertex/create_endpoint.sh etude-noop
./vertex/deploy_model.sh etude-noop
./vertex/deploy_endpoint_model.sh etude-noop etude-noop

# run loadtest
# ...

# cleanup resources
#./vertex/purge_endpoint.sh etude-noop
#./vertex/purge_model.sh etude-noop
