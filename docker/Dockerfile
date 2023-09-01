FROM spack/ubuntu-jammy:latest
RUN spack install intel-oneapi-compilers@2022.1.0 && \
    spack load intel-oneapi-compilers && \
    spack compiler find && \
    spack unload intel-oneapi-compilers
ENV NESO_ROOT=/root/NESO
RUN git clone --branch mmg/updated-spec --single-branch --recurse-submodules -j4 \
    https://github.com/matt-graham/NESO.git $NESO_ROOT
ENV I_MPI_FABRICS=shm
ENV SYCL_DEVICE_FILTER=host
RUN spack env activate -d $NESO_ROOT && \
    spack install neso%oneapi
ENV ENTRYPOINT=/root/entrypoint.sh
RUN echo "#! /usr/bin/env bash" >> $ENTRYPOINT && \
    echo "source $SPACK_ROOT/share/spack/setup-env.sh" >> $ENTRYPOINT && \
    echo "unset CURRENTLY_BUILDING_DOCKER_IMAGE" >> $ENTRYPOINT && \
    echo "spack env activate -d $NESO_ROOT" >> $ENTRYPOINT && \
    echo "spack load intel-oneapi-mpi" >> $ENTRYPOINT && \
    echo "export PATH=$PATH:$NESO_ROOT/views/oneapi-dpcpp/bin/" >> $ENTRYPOINT && \
    echo "exec \"\$@\"" >> $ENTRYPOINT && \
    chmod +x $ENTRYPOINT
ENTRYPOINT $ENTRYPOINT
