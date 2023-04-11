### TODO's for the load test on SBR Neural models

### Infra
* Add Frank to recoanalytics project
* Can we run/coordinate the test flow without gitlab as a dependency?

### Model
* Add code to train a model
    * For a catalog of size 'C' and 'runtime'
* Create a Docker image using a public base image
* Deploy the model
   * Deploy the Docker image
   * Deploy an Vertex-AI endpoint
   * Deploy the image for the endpoint
* Add code for a service-layer to be used in the TorchServe
* Determine how to remove the deployed model


### Load generator
* Create a Docker image using a public base image
* Introduce command line parameters/config file to control the test.
   * 'C' catalog size
   * Vertex AI Endpoint ID
   * Google cloud project
* Create kubernetes config to deploy the loadtest
* Add code to copy the result from loadtest to storage bucket
* Code to coordinate the whole ensemble.
