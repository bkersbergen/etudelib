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
	http://localhost:7080/v1/recommend \
	--data "{\"instances\": [{\"context\": [1, 2, 3]}],\"parameters\": [{\"runtime\":  \"\"}]}"

buildserving:  ## build the serving application for the models
	docker buildx build --platform linux/amd64 -t base -f docker/BaseDockerfile .

interactiveserving:  ## run the docker base image interactive
	docker run --rm -it --platform linux/amd64 -v $(shell pwd):/local/ -i -t base /bin/bash

buildmodel:  ## build a model
	docker build --platform linux/amd64 \
		--build-arg path_to_model="model_store/noop_bolcom_c1000000_t50_jitopt.pth" \
		--build-arg path_to_payload="model_store/noop_bolcom_c1000000_t50_payload.yaml" \
		-t noop \
		-f docker/ModelDockerfile .

interactivemodel:  ## run the model interactive
	docker run --rm -it --platform linux/amd64 -v $(shell pwd):/local/ -i -t noop /bin/bash

runmodel:  ## run the model
	docker run --platform linux/amd64 --expose 7080 -t noop

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

