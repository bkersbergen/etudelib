SHELL:=/bin/bash
.DEFAULT_GOAL:=help
PWD:=$(shell pwd)
USER ?= -SA
PROJECT_ID=bk47479

hello:
	@echo "Hello"

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
	docker build --platform linux/amd64 -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest -f docker/BaseDockerfile .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

serve_interactive:  ## run the docker base image interactive
	docker run --rm -it --entrypoint /bin/bash --platform linux/amd64 --user root -v $(shell pwd):/local/ -i -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

serve_deploy:
	echo needs implementation

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

