SHELL:=/bin/bash
.DEFAULT_GOAL:=help

PROJECT_ID=bk474718
REGION="europe-west4"
USER ?= -SA
JOB_NAME := $(USER)_etude_microbenchmark_$(shell date +'%Y%m%d_%H%M%S')
IMAGE_URI_MICROBENCHMARK=eu.gcr.io/$(PROJECT_ID)/etudelib/etudelib_microbenchmark:latest
IMAGE_URI_TORCHSERVE=eu.gcr.io/$(PROJECT_ID)/etudelib/etudelib_torchserve:latest

.PHONY: *
# .PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))

docker: ## build and push the docker images
	$(MAKE) serving_buildpush
	$(MAKE) loadgenerator_buildpush
	$(MAKE) training_buildpush
	$(MAKE) microbenchmark_build

docker_prune: ## prune docker to free up resources
	docker system prune -a -f
	docker image prune -a -f

run_end_to_end_benchmark:  ## execute the deployed benchmark pipeline
	.k8s/experiment_pipeline.sh $(PROJECT_ID)

infra:  ## Create the infrastructure in GCP
	infra/create_infra.sh $(PROJECT_ID)

serving_buildpush:  ## build the serving application and push it to the docker repo
	docker build --platform linux/amd64 -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest -f .docker/rust-serving.Dockerfile .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest



loadgenerator_buildpush:  ## build the serving application and push it to the docker repo
	docker build --platform linux/amd64 --build-arg PARENT_IMAGE="azul/zulu-openjdk-debian:17-latest" --tag "eu.gcr.io/$(PROJECT_ID)/etude-loadgen:latest" -f .docker/loadgen.Dockerfile .
	docker push "eu.gcr.io/$(PROJECT_ID)/etude-loadgen:latest"

training_buildpush:  ## build the models training application and push it to the docker repo
	docker build --platform linux/amd64 --build-arg PARENT_IMAGE="eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest" -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_modeltraining:latest -f .docker/training.Dockerfile .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_modeltraining:latest

training_k8s_deploy_gpu:  ## deploy the model training application in kubernetes
	YAML_TEMPLATE=.k8s/etudelib-training_gpu.yaml; \
	$(MAKE) undeploy_training_run; \
    kubectl apply -f <( \
        sed -e 's/$${PROJECT_ID}/$(PROJECT_ID)/' \
            $$YAML_TEMPLATE \
    );

undeploy_training_run:  ## undeploys training_run from kubernetes
	-kubectl delete job etudelibtraining

microbenchmark_build: ## Build and push the microbenchmark image to the repository.
	docker build --platform linux/amd64 --build-arg PARENT_IMAGE="eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest" -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_microbenchmark:latest -f .docker/microbenchmark.Dockerfile .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_microbenchmark:latest

undeploy_microbenchmark_run:  ## undeploys microbenchmark from kubernetes
	-kubectl delete job etudelibmicrobenchmark; \

microbenchmark_run: ## Run the microbenchmark in Google AI platform
	YAML_TEMPLATE=.k8s/etudelib-microbenchmark_gpu.yaml; \
	$(MAKE) undeploy_microbenchmark_run; \
    kubectl apply -f <( \
        sed -e 's/$${PROJECT_ID}/$(PROJECT_ID)/' \
        -e 's/$${PROJECT_ID}-shared/$(PROJECT_ID)-shared/' \
            $$YAML_TEMPLATE \
    );

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

