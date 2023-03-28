# Guide to Setting up the CI using the Docker images

## Steps

1. Build the docker image using the Dockerfile in the .ci directory.
   Make sure you are in the root directory of `etudelib`.
   `docker build -f .ci/Dockerfile -t test .`

2. Run docker image
`docker run test`