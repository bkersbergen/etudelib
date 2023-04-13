SHELL:=/bin/bash
.DEFAULT_GOAL:=help

PROJECT=bolcom-pro-reco-analytics-fcc
REGION="europe-west4"
USER ?= -SA
JOB_NAME := $(USER)_etude_microbenchmark_$(shell date +'%Y%m%d_%H%M%S')
IMAGE_URI_MICROBENCHMARK=eu.gcr.io/$(PROJECT)/etudelib_microbenchmark:latest
IMAGE_URI_TORCHSERVE=eu.gcr.io/$(PROJECT)/etudelib_torchserve:latest

microbenchmark_build: ## Build and push the microbenchmark image to the repository.
	@docker build -t $(IMAGE_URI_MICROBENCHMARK) -f .ci/Dockerfile .
	@docker push $(IMAGE_URI_MICROBENCHMARK)

microbenchmark_run: ## Run the microbenchmark in Google AI platform
	@gcloud beta ai-platform jobs submit training $(JOB_NAME) \
	  --region $(REGION) \
	  --project $(PROJECT) \
	  --master-image-uri $(IMAGE_URI_MICROBENCHMARK) \
	  --scale-tier CUSTOM \
	  --master-machine-type n1-highmem-8 \
	  --master-accelerator count=1,type=nvidia-tesla-t4

torchserve_baseimage_build: ## Build and push the Docker base image that the deployed models use.
	@cp requirements/base.txt .docker/requirements.txt && cd .docker && docker build -t $(IMAGE_URI_TORCHSERVE) -f ServingDockerfile .
	@docker push $(IMAGE_URI_TORCHSERVE)

torchserve_model_build: ## Build and push the Docker image for the CORE algorithm
	@cd .docker && docker build -t core -f ModelDockerfile .

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

