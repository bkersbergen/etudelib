FROM pytorch/torchserve:0.8.1-gpu

USER 0

RUN set -eux; \
    apt-get update; \
    apt-get install -y \
    curl \
    gnupg2 \
    wget \
    apt-transport-https;

# Install Google SDK
RUN apt-get install -y apt-transport-https ca-certificates gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

WORKDIR /home/model-server

COPY model.mar /home/model-server/model-store
COPY start.sh /home/model-server

RUN chown -R model-server:model-server ./start.sh
RUN chmod +x ./start.sh

USER model-server
ENTRYPOINT ["./start.sh"]