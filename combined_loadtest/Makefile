SHELL:=/bin/bash
.DEFAULT_GOAL:=help

PROJECT=bk47475
REGION="europe-west4"
USER ?= -SA
JOB_NAME := $(USER)_etude_deployedbenchmark_$(shell date +'%Y%m%d_%H%M%S')
IMAGE_URI_DEPOYEDBENCHMARK=eu.gcr.io/$(PROJECT)/etudelib/etudelib_deployedbenchmark:latest

depoyedbenchmark_build_push:  ## Push docker image to playground
	docker build --platform linux/amd64 -t eu.gcr.io/$(PROJECT)/etudelib/etudelib_torchserve:latest -f Dockerfile .
	docker push eu.gcr.io/$(PROJECT)/etudelib/etudelib_torchserve:latest


depoyedbenchmark_run_locally: ## Build and push the microbenchmark image to the repository.
	docker run \
		-e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/legacy_credentials/${USER}@bol.com/adc.json \
		-e USER=${USER} \
		-v ~/.config/gcloud:/app/.config/gcloud \
		--platform linux/amd64 \
		-it \
		-p 7080:7080 -p 7081:7081 \
		eu.gcr.io/$(PROJECT)/etudelib/etudelib_torchserve:latest \
		gs://$(PROJECT)-shared/model_store/noop_bolcom_c1000000_t50_jitopt/model.mar
		#\
		#--entrypoint /bin/bash




help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)


