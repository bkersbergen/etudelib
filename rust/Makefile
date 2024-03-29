SHELL:=/bin/bash
.DEFAULT_GOAL:=help
PWD:=$(shell pwd)
USER ?= -SA
PROJECT_ID=bk474715

hello:
	@echo "Hello"

deploy_cpu_static:  ## deploy dummy model 2vCPU 2GB mem
	YAML_TEMPLATE=etudelibrust-deployment_static_baseline_2cpu.yaml; \
	$(MAKE) undeploy_serving; \
    kubectl apply -f <( \
        sed -e 's/$${PROJECT_ID}/$(PROJECT_ID)/' \
            $$YAML_TEMPLATE \
    ); \
	kubectl apply -f etudelibrust-service.yaml

deploy_loadgen_static:
	./deploy_loadgen.sh bk474714 "http://34.118.227.233:8080/predictions/model/1.0/" 1000 gs://bk474714-shared/static_baseline2vcpu 1000 10


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
	docker build --no-cache --platform linux/amd64 -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest -f docker/BaseDockerfile .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

deploy_cpu_serving:  ## deploy rust serving engine in kubernetes
	YAML_TEMPLATE=etudelibrust-deployment_cpu.yaml; \
	$(MAKE) undeploy_serving; \
    kubectl apply -f <( \
        sed -e 's/$${PROJECT_ID}/$(PROJECT_ID)/' \
            $$YAML_TEMPLATE \
    ); \
	kubectl apply -f etudelibrust-service.yaml

deploy_gpu_serving:  ## deploy rust serving engine in kubernetes
	YAML_TEMPLATE=etudelibrust-deployment_gpu.yaml; \
	$(MAKE) undeploy_serving; \
    kubectl apply -f <( \
        sed -e 's/$${PROJECT_ID}/$(PROJECT_ID)/' \
            $$YAML_TEMPLATE \
    ); \
	kubectl apply -f etudelibrust-service.yaml


undeploy_serving:  ## undeploys etudelibrust from kubernetes
	-kubectl delete deployment etudelibrust

train:  ## train and deploy the models to the storage bucket
	. venv/bin/activate; python train.py $(PROJECT_ID)


serve_interactive:  ## run the docker base image interactive
	docker run --rm -it --entrypoint /bin/bash --platform linux/amd64 --user root -v $(shell pwd):/local/ -i -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

