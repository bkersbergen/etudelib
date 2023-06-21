SHELL:=/bin/bash
.DEFAULT_GOAL:=help
PWD:=$(shell pwd)
USER ?= -SA
PROJECT_ID=bk47476

hello:
	@echo "Hello"

run_local: ## Build and run the Rust application in debug mode
	cargo run --release --bin serving -- $(ARGS)

example: ## simple post request to endpoint
	curl -X POST -H \
	"Content-Type: application/json" \
	http://localhost:8080/predictions/model \
	--data "{\"instances\": [{\"context\": [1, 2, 3]}],\"parameters\": [{\"runtime\":  \"\"}]}"

buildpush:  ## build the serving application for the models
	docker build --platform linux/amd64 -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest -f docker/BaseDockerfile .
	docker push eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

serve_interactive:  ## run the docker base image interactive
	docker run --rm -it --entrypoint /bin/bash --platform linux/amd64 --user root -v $(shell pwd):/local/ -i -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

model_interactive:  ## run the model interactive
	docker run --rm -it --platform linux/amd64 -v $(shell pwd):/local/ -i -t eu.gcr.io/$(PROJECT_ID)/etudelib/noop_bolcom_c1000000_t50_jitopt:latest /bin/bash

model_run:  ## run the model
	docker run --platform linux/amd64 -p 8080:8080 -t eu.gcr.io/$(PROJECT_ID)/etudelib/serving_rust:latest

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

