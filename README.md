# ETUDELIB

Etude - A Framework for Measuring the Inference Latency of Session-Based Recommendation Models at Scale


## Introduction
Etudelib aims to enable algorithm designers and data scientists to evaluate the computational performance of their models

### Key Features
* Ready to use deep learning SBR algorithms and benchmarks
* PyTorch Lightning based model implementations to reduce the boilerplate code and limit the implementation efforts to the bare essentials
* A set of tools for evaluate the computational performance of (their) models
* A set of inference tools to quick and easy deployment of the standard or custom session based recommendation models.

### Getting started
To get an overview of all the devices where `etudelib` has been tested thoroughly, look at the hardware section in the documention

#### Local install
It is highly recommended to use a virtual environment when installing etudelib. For instance as:

```bash
git clone git@github.com:bkersbergen/etudelib.git
cd etudelib
pip install -e .
```


### Load test
Always use the same machine for models: 
* model names: 'core', 'gcsan', 'gru4rec', 'lightsans', 'narm', 'repeatnet', 'sasrec', 'sine', 'srgnn', 'stamp'
* catalog size: [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 20_000_000]
* runtimes: eager, jitopt
* device_types: cpu or gpu 


## Training 

Training a model on a specific dataset and category requires further configuration. Each model has its own configuration file, config.yaml , which contains data, model and training configurable parameters. To train a specific model on a specific dataset and category, the config file is to be provided:
`python etudelib/tools/train.py --config <path/to/model/config.yaml>`

Alternatively, a model name could also be provided as an argument, where the scripts automatically find the corresponding config file.
where the current available models are:



#### Troubleshooting install
Mac M1 grpcio install error: "Python.h: No such file or directory"
This can be solved by pip install special fixed builds
```bash
pip install https://github.com/pietrodn/grpcio-mac-arm-build/releases/download/1.51.1/grpcio-1.51.1-cp39-cp39-macosx_11_0_arm64.whl
```
