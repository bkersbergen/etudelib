FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel AS devel
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    curl \
	llvm \
    clang;

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo search
