FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime AS runtime
RUN set -eux; \
    apt-get update; \
    apt-get install -y \
    curl \
    gnupg2 \
    wget \
    apt-transport-https;

# Setup the local account
RUN groupadd -r app && useradd -r -g app -u 1001 app --home /app
RUN mkdir -p /app/etudelib
RUN chown -R app:app /app
USER app

WORKDIR /app
COPY --chown=app:app . /app/

RUN pip install -e .

ENTRYPOINT ["python", "etudelib/tools/microbenchmarking.py"]

