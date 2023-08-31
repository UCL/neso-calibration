FROM spack/ubuntu-jammy:latest as builder
ENV NESO_ROOT=/root/NESO
RUN spack install intel-oneapi-compilers@2022.1.0 && \
    spack load intel-oneapi-compilers && \
    spack compiler find && \
    spack unload intel-oneapi-compilers
RUN git clone --branch mmg/updated-spec --single-branch --recurse-submodules -j4 https://github.com/matt-graham/NESO.git $NESO_ROOT
WORKDIR $NESO_ROOT
RUN . activate && spack install
