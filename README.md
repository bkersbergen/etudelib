# ETUDELIB

Etude - A Framework for Measuring the Inference Latency of Session-Based Recommendation Models at Scale


### Introduction
Etudelib aims to enable algorithm designers and data scientists to evaluate the computational performance of their models

### Getting started
To get an overview of all the devices where `etudelib` has been tested thoroughly, look at the hardware section in the documention

## Experiments
[documentation on experiments](experiments.md)






#### Troubleshooting install
Mac M1 grpcio install error: "Python.h: No such file or directory"
This can be solved by pip install special fixed builds
```bash
pip install https://github.com/pietrodn/grpcio-mac-arm-build/releases/download/1.51.1/grpcio-1.51.1-cp39-cp39-macosx_11_0_arm64.whl
```
