SHELL:=/bin/bash
.DEFAULT_GOAL:=help
PWD:=$(shell pwd)
USER ?= -SA

export LIBTORCH=$(shell python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib

configure: requirements.txt ## configure a virtual environment with pytorch for its c++ libs
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

train: ## Train a PyTorch model and persist it to disk
	python train.py

run: ## Build and run the Rust application in debug mode
	cargo run -- models/mnist_cpu.pt

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

