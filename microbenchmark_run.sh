#!/bin/bash
set -u # error on undefined variables

# Configurable parameters
PROJECT="bolcom-pro-reco-analytics-fcc"
REGION="europe-west4"
USER=${USER:-SA}

IMAGE_URI=eu.gcr.io/${PROJECT}/etudelib-microbenchmark:main
JOB_NAME=${USER}_etudelib_$(date +%Y%m%d_%H%M%S)
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --project $PROJECT \
  --master-image-uri $IMAGE_URI \
  --scale-tier CUSTOM \
  --master-machine-type n1-highmem-8 \
  --master-accelerator count=1,type=nvidia-tesla-t4 \


exit


#!/usr/bin/env bash

# set -x # set debug on

read -p "You are about to evaluate 800 models. Are you sure? Press 'c' to continue" -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Cc]$ ]]
then
    echo You pressed $REPLY and not 'c'
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

# Hyper parameter optimization for a selection of algorithms
# The hyper parameter optimization is done by starting each algorithm a 100 times
# For each submission it will choose a random combination of parameters.

# Configurable parameters
PROJECT="bolcom-pro-reco-analytics-fcc"
REGION="europe-west4"
USER=${USER:-SA}

# IMAGE_URI: the complete URI location for Cloud Container Registry
export IMAGE_URI=eu.gcr.io/${PROJECT}/tinytoolbox-gpu:master

function do_train_evaluate_for_algorithms(){
  echo start do_train_evaluate_for_algorithms
  for ddate in 2020-06-04; do
#    for algorithm in ar gru4rec sknn sr vsknn narm; do
    for algorithm in vsknn; do
      # run algorithms that do not support GPU on infra without GPU
      # we use tensorflow-gpu version because then we only need one image. However this tensorflow will crash without GPU.
      export JOB_NAME=${USER}_${algorithm}_${ddate//-/_}_train_eval_cpu_$(date +%Y%m%d_%H%M%S)
      echo ${JOB_NAME}

      gcloud beta ai-platform jobs submit training $JOB_NAME \
        --region $REGION \
        --project $PROJECT \
        --master-image-uri $IMAGE_URI \
        --scale-tier CUSTOM \
        --master-machine-type n1-highmem-8 \
        -- \
        --config_path=conf/train_eval_multiple_56m/${algorithm}_${ddate}.yml \
        --use_gcp=True
    done

#    for algorithm in stamp; do
#      # run algorithms that do support GPU on infra with GPU
#      export JOB_NAME=${USER}_${algorithm}_${ddate//-/_}_train_eval_gpu_$(date +%Y%m%d_%H%M%S)
#      echo ${JOB_NAME}
#
#      gcloud beta ai-platform jobs submit training $JOB_NAME \
#        --region $REGION \
#        --project $PROJECT \
#        --master-image-uri $IMAGE_URI \
#        --scale-tier CUSTOM \
#        --master-machine-type n1-highmem-8 \
#        --master-accelerator count=1,type=nvidia-tesla-t4 \
#        -- \
#        --config_path=conf/train_eval_multiple_56m/${algorithm}_${ddate}.yml \
#        --use_gcp=True
#    done
  done
  echo end echo start do_train_evaluate_for_algorithms
}

do_train_evaluate_for_algorithms

