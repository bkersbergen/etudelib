FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Inspired by https://github.com/Rust-GPU/Rust-CUDA/blob/master/Dockerfile which is no longer maintained

# Update default packages
RUN apt-get update

# https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image
# Get Ubuntu packages
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    curl xz-utils pkg-config libssl-dev zlib1g-dev libtinfo-dev libxml2-dev llvm clang zip less vim


# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

# Install TorchLib C++ with latest CUDA support (/libtorch)
RUN curl https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip -o libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
RUN rm libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip

# set env
ENV LLVM_CONFIG=/usr/bin/llvm-config
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=$CUDA_ROOT
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=error
ENV PATH=$CUDA_ROOT/nvvm/lib64:/root/.cargo/bin:$PATH

# make ld aware of necessary *.so libraries
RUN echo $CUDA_ROOT/lib64 >> /etc/ld.so.conf &&\
    echo $CUDA_ROOT/compat >> /etc/ld.so.conf &&\
    echo $CUDA_ROOT/nvvm/lib64 >> /etc/ld.so.conf &&\
    echo /libtorch/lib >> /etc/ld.so.conf &&\
    ldconfig

# Install Google SDK
RUN apt-get install -y apt-transport-https ca-certificates gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

## Install PyTorch with latest CUDA support
RUN apt-get install -y python3-pip
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir omegaconf scipy lightning

COPY ./rust/src src
COPY ./rust/Cargo.toml ./
COPY ./rust/build_loop.sh ./

# Trigger Onnx to download CUDA version
ENV ORT_USE_CUDA=1
RUN chmod +x ./build_loop.sh ; ./build_loop.sh > build.log
ENV LIBTORCH=/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:/usr/local/lib:$LD_LIBRARY_PATH

# A simple `cargo build` failed over and over again. Randomly producing all kinds of compile- or out of memory- errors
# I was not able to pin point it. Suggestions from the internet did not help consistently.
# Perhaps its related by my M1 emulating a x64 on QEMU.
# The best way to make it work is keep compiling until its successful. This seems to work on all platforms.

COPY ./rust/config config
RUN mkdir model_store
COPY ./rust/start.sh ./
COPY ./rust/predict_by_id.sh ./
RUN chmod +x ./*.sh

# free up disk space
RUN apt-get autoclean

ENTRYPOINT [ "./start.sh" ]
EXPOSE 8080/tcp