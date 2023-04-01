SHELL:=/bin/bash
.DEFAULT_GOAL:=help

PROJECT=bolcom-pro-reco-analytics-fcc
REGION="europe-west4"
USER ?= -SA
JOB_NAME := $(USER)_etude_microbenchmark_$(shell date +'%Y%m%d_%H%M%S')
IMAGE_URI=eu.gcr.io/$(PROJECT)/etude_microbenchmark:latest

image: ## Build and push the microbenchmark image to the repository.
	@docker build -t $(IMAGE_URI) -f .ci/Dockerfile .
	@docker push $(IMAGE_URI)

microbenchmark: ## Run the microbenchmark in Google AI platform
	@gcloud beta ai-platform jobs submit training $(JOB_NAME) \
	  --region $(REGION) \
	  --project $(PROJECT) \
	  --master-image-uri $(IMAGE_URI) \
	  --scale-tier CUSTOM \
	  --master-machine-type n1-highmem-8 \
	  --master-accelerator count=1,type=nvidia-tesla-t4 \

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)
