SHELL:=/bin/bash
.DEFAULT_GOAL:=help
PWD:=$(shell pwd)
USER ?= -SA
PROJECT_ID=bk47472

hello:
	@echo "Hello"

configure: requirements.txt ## configure a virtual environment with pytorch for its c++ libs
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

serving: ## Build and run the Rust application in debug mode
	cargo run --release --bin serving -- $(ARGS)

example: ## simple post request to endpoint
	curl -X POST -H \
	"Content-Type: application/json" \
	http://localhost:8080/predictions/model/1.0/ \
	--data "{\"instances\": [{\"context\": [1, 2, 3]}],\"parameters\": [{\"runtime\":  \"\"}]}"

serve_buildpush:  ## build the serving application for the models
	docker build --platform linux/amd64 -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest -f docker/BaseDockerfile .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

deploy_cpu_serving:  ## deploy rust serving engine in kubernetes
	$(MAKE) undeploy_serving
	kubectl apply -f etudelibrust-deployment_cpu.yaml
	kubectl apply -f etudelibrust-service.yaml

deploy_gpu_serving:  ## deploy rust serving engine in kubernetes
	$(MAKE) undeploy_serving
	kubectl apply -f etudelibrust-deployment_gpu.yaml
	kubectl apply -f etudelibrust-service.yaml


undeploy_serving:  ## undeploys etudelibrust from kubernetes
	-kubectl delete deployment etudelibrust

train:  ## train and deploy the models to the storage bucket
	. venv/bin/activate; python train.py $(PROJECT_ID)


serve_interactive:  ## run the docker base image interactive
	docker run --rm -it --entrypoint /bin/bash --platform linux/amd64 --user root -v $(shell pwd):/local/ -i -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

