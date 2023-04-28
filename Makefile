SHELL:=/bin/bash
.DEFAULT_GOAL:=help

PROJECT=bolcom-pro-reco-analytics-fcc
REGION="europe-west4"
USER ?= -SA
JOB_NAME := $(USER)_etude_microbenchmark_$(shell date +'%Y%m%d_%H%M%S')
IMAGE_URI_MICROBENCHMARK=eu.gcr.io/$(PROJECT)/etudelib/etudelib_microbenchmark:latest
IMAGE_URI_TORCHSERVE=eu.gcr.io/$(PROJECT)/etudelib/etudelib_torchserve:latest

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

model_baseimage: ## Build and push the Docker base image that the deployed models use.
	@cp requirements/base.txt .docker/requirements.txt && docker build -t $(IMAGE_URI_TORCHSERVE) -f .docker/ServingDockerfile .
	@docker push $(IMAGE_URI_TORCHSERVE)

model_build: ## Build and push a marfile Docker image.
	@test $(MARFILE_WO_EXT) || ( echo ">> MARFILE_WO_EXT must be specified. E.g. make model_build MARFILE_WO_EXT=core_bolcom_c100000_t50_eager"; exit 1 )
	@cd .docker && \
		docker build --platform=linux/amd64 --build-arg MODELFILE_WO_EXT=$(MARFILE_WO_EXT) -t eu.gcr.io/$(PROJECT)/etudelib/$(MARFILE_WO_EXT):latest -f ModelDockerfile . && \
		docker push eu.gcr.io/$(PROJECT)/etudelib/$(MARFILE_WO_EXT):latest

model_run:  ## Run marfile Docker image locally
	@test $(MARFILE_WO_EXT) || ( echo ">> MARFILE_WO_EXT must be specified. E.g. make torchserve_run MARFILE_WO_EXT=core_bolcom_c100000_t50_eager"; exit 1 )
	@docker run --platform linux/amd64 -p 7080:7080 -p 7081:7081 eu.gcr.io/$(PROJECT)/etudelib/$(MARFILE_WO_EXT):latest

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

