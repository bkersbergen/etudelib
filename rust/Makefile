SHELL:=/bin/bash
.DEFAULT_GOAL:=help
PWD:=$(shell pwd)
USER ?= -SA

hello:
	@echo "Hello"

ifeq (,$(filter defined,$(origin LIBTORCH)))
	@if [ -f .libtorch_path ]; then \
		LIBTORCH=$$(cat .libtorch_path); \
	else \
		LIBTORCH=$$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)'); \
		echo $$LIBTORCH > .libtorch_path; \
	fi
	@DYLD_LIBRARY_PATH=$$LIBTORCH/lib
	@export LIBTORCH
	@export DYLD_LIBRARY_PATH
endif

configure: requirements.txt ## configure a virtual environment with pytorch for its c++ libs
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

train: ## Train a PyTorch model and persist it to disk
	python train.py

serving: ## Build and run the Rust application in debug mode
	cargo run --release --bin serving -- $(ARGS)

example: ## simple post request to endpoint
	curl -X POST -H \
	"Content-Type: application/json" \
	http://localhost:7080/predictions/model \
	--data "{\"instances\": [{\"context\": [1, 2, 3]}],\"parameters\": [{\"runtime\":  \"\"}]}"

serve_buildpush:  ## build the serving application for the models
	docker build --platform linux/amd64 -t eu.gcr.io/bolcom-pro-reco-analytics-fcc/etudelib/serving_rust:latest -f docker/BaseDockerfile .
	docker push eu.gcr.io/bolcom-pro-reco-analytics-fcc/etudelib/serving_rust:latest

serve_interactive:  ## run the docker base image interactive
	docker run --rm -it --entrypoint /bin/bash --platform linux/amd64 --user root -v $(shell pwd):/local/ -i -t eu.gcr.io/bolcom-pro-reco-analytics-fcc/etudelib/serving_rust:latest

model_buildpush:  ## build and push model
	docker build --platform linux/amd64 \
		--build-arg path_to_model="model_store/noop_bolcom_c1000000_t50_jitopt.pth" \
		--build-arg path_to_payload="model_store/noop_bolcom_c1000000_t50_payload.yaml" \
		--build-arg path_to_config="config/serving.yaml" \
		-t eu.gcr.io/bolcom-pro-reco-analytics-fcc/etudelib/noop_bolcom_c1000000_t50_jitopt:latest \
		-f docker/ModelDockerfile .
	docker push eu.gcr.io/bolcom-pro-reco-analytics-fcc/etudelib/noop_bolcom_c1000000_t50_jitopt:latest

model_interactive:  ## run the model interactive
	docker run --rm -it --platform linux/amd64 -v $(shell pwd):/local/ -i -t eu.gcr.io/bolcom-pro-reco-analytics-fcc/etudelib/noop_bolcom_c1000000_t50_jitopt:latest /bin/bash

model_run:  ## run the model
	docker run --platform linux/amd64 -p 7080:7080 -t eu.gcr.io/bolcom-pro-reco-analytics-fcc/etudelib/noop_bolcom_c1000000_t50_jitopt:latest

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

