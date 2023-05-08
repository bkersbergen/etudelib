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

run: ## Build and run the Rust application in debug mode
	cargo run -- model_store/noop_bolcom_c10000_t50_jitopt.pth

example: ## simple post request to endpoint
	curl -X POST \
	-H 'Content-Type: application/json' \
	http://localhost:7080/v1/recommend  \
	--data '{"item_ids": [1,5,3,1,7,1], "session_id": "abcdefg"}'

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

