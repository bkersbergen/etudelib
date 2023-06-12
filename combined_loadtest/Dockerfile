FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime AS runtime
RUN set -eux; \
    apt-get update; \
    apt-get install -y \
    curl \
    gnupg2 \
    wget \
    apt-transport-https;


# Install Eclipse adoptium Java
RUN mkdir -p /etc/apt/keyrings
RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc
RUN echo "deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update -y && apt-get install -y temurin-17-jdk

# Install Google SDK
RUN apt-get install -y apt-transport-https ca-certificates gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

WORKDIR /app
RUN mkdir -p models/

# Install generic configuration
COPY log4j2.xml ./conf/
COPY torchserve.properties ./conf/
COPY start.sh ./
RUN chmod +x ./start.sh
COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN echo Using hardcoded model in the image, not using the storage bucket atm until we fixed read storage bucket permission
COPY model.mar /app/models/

# Setup the local account
RUN groupadd -r app && useradd -r -g app -u 1001 app --home /app
RUN chown -R app:app /app
USER app

EXPOSE 7080 7081 7082

ENTRYPOINT ["./start.sh"]


# WORKDIR /app
# RUN pip install torchserve numpy
#
# # Installs python dependencies.
# COPY .docker/requirements.txt /tmp/
# RUN pip install -r /tmp/requirements.txt
#
# # Install generic configuration
# COPY .docker/log4j2.xml ./conf/
# COPY .docker/torchserve.properties ./conf/
# COPY .docker/start.sh ./
# RUN chmod +x ./start.sh
#
# COPY etudelib/ ./etudelib/
# COPY setup.py ./
# COPY pyproject.toml ./
# COPY README.md ./
# COPY requirements/ ./requirements/
# RUN pip install -e .
