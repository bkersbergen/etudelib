SHELL:=/bin/bash
.DEFAULT_GOAL:=help

PROJECT_ID=bk47473
REGION="europe-west4"
USER ?= -SA
JOB_NAME := $(USER)_etude_microbenchmark_$(shell date +'%Y%m%d_%H%M%S')
IMAGE_URI_MICROBENCHMARK=eu.gcr.io/$(PROJECT_ID)/etudelib/etudelib_microbenchmark:latest
IMAGE_URI_TORCHSERVE=eu.gcr.io/$(PROJECT_ID)/etudelib/etudelib_torchserve:latest

docker_prune: ## prune docker to free up resources
	docker system prune -a -f
	docker image prune -a -f


infra:  ## Create the infrastructure in GCP
	@infra/create_infra.sh $(PROJECT_ID)

rustserving_buildpush:  ## build the serving application for the models
	docker build --no-cache --platform linux/amd64 -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest -f .docker/rust-serving.Dockerfile .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

loadgenerator_build_push:  ## build the deployed load generator for the models
	docker build --platform linux/amd64 --build-arg PARENT_IMAGE="azul/zulu-openjdk-debian:17-latest" --tag "eu.gcr.io/$(PROJECT_ID)/etude-loadgen:latest" -f .docker/loadgen.Dockerfile .
	docker push "eu.gcr.io/$(PROJECT_ID)/etude-loadgen:latest"


training_buildpush:  ## build the serving application for the models
	docker build --no-cache --platform linux/amd64 -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_modeltraining:latest -f .ci/DockerfileModelTraining .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_modeltraining:latest

training_run_gpu:  ## deploy rust serving engine in kubernetes
	YAML_TEMPLATE=.ci/etudelib-training_gpu.yaml; \
	$(MAKE) undeploy_training_run; \
    kubectl apply -f <( \
        sed -e 's/$${PROJECT_ID}/$(PROJECT_ID)/' \
            $$YAML_TEMPLATE \
    );

undeploy_training_run:  ## undeploys training_run from kubernetes
	-kubectl delete deployment etudelibtraining

microbenchmark_build: ## Build and push the microbenchmark image to the repository.
	@docker build -t $(IMAGE_URI_MICROBENCHMARK) -f .docker/microbenchmark.Dockerfile .
	@docker push $(IMAGE_URI_MICROBENCHMARK)

microbenchmark_run: ## Run the microbenchmark in Google AI platform
	@gcloud beta ai-platform jobs submit training $(JOB_NAME) \
	  --region $(REGION) \
	  --project $(PROJECT_ID) \
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
		docker build --platform=linux/amd64 --build-arg MODELFILE_WO_EXT=$(MARFILE_WO_EXT) -t eu.gcr.io/$(PROJECT_ID)/etudelib/$(MARFILE_WO_EXT):latest -f ModelDockerfile . && \
		docker push eu.gcr.io/$(PROJECT_ID)/etudelib/$(MARFILE_WO_EXT):latest

model_run:  ## Run marfile Docker image locally
	@test $(MARFILE_WO_EXT) || ( echo ">> MARFILE_WO_EXT must be specified. E.g. make torchserve_run MARFILE_WO_EXT=core_bolcom_c100000_t50_eager"; exit 1 )
	@docker run --platform linux/amd64 -p 7080:7080 -p 7081:7081 eu.gcr.io/$(PROJECT_ID)/etudelib/$(MARFILE_WO_EXT):latest

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

