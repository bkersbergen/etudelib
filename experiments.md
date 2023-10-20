# ETUDE experiments




### Prepare infrastructure
Etude can automatically setup the cloud infrastructure. This is a one time operation.
Edit the Makefile and change the PROJECT_ID to match your google cloud name.
Then execute from a terminal:

`make infra`

Prepare the Docker images:

`make docker`

## Getting started 
Prepare all models and save them to a storage bucket 

`make training_k8s_deploy_gpu`

Execute the end-to-end benchmark:

`make run_end_to_end_benchmark`

To execute the microbenchmark:

`make microbenchmark_run`
