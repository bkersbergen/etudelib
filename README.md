# ETUDELIB

Etude - A Framework for Measuring the Inference Latency of Session-Based Recommendation Models at Scale


## Introduction
Etudelib is a deep learning library that aims to enable algorithm designers and data scientists to evaluate the computational performance of their models, without forcing them to go through a difficult deployment cycle

### Key Features
* Ready to use deep learning SBR algorithms and benchmarks
* PyTorch Lightning based model implementations to reduce the boilerplate code and limit the implementation efforts to the bare essentials
* All models can be exported for accelerated inference hardware
* A set of tools for quick and easy evaluate the computational performance of (their) models, without forcing them to go through a difficult deployment cycle
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



## training 

Training a model on a specific dataset and category requires further configuration. Each model has its own configuration file, config.yaml , which contains data, model and training configurable parameters. To train a specific model on a specific dataset and category, the config file is to be provided:
`python tools/train.py --config <path/to/model/config.yaml>`

Alternatively, a model name could also be provided as an argument, where the scripts automatically find the corresponding config file.
where the current available models are:

TODO: list of models


For example, to train LightSans you can use
xxxxx


## Inference
xxxxx


## Exporting Model to ONNX
It is possible to export your model to ONNX 
```
optimization:
    export_mode: "onnx"
```

## Hyper parameter Optimization
To run hyperparameter optimization, use the following command:

```bash
python tools/hpo/sweep.py \
    --model lightsans
    --model_config path_to_config.yaml
    --sweep_config tools/hpo/sweep.yaml
```


