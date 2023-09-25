FROM spack/ubuntu-jammy:latest
ENV NESO_ROOT=/root/NESO
RUN git clone --branch mmg/updated-spec --single-branch --recurse-submodules -j4 \
    https://github.com/matt-graham/NESO.git $NESO_ROOT
ENV SYCL_DEVICE_FILTER=host
RUN spack env activate -d $NESO_ROOT && \
    spack install neso%gcc
ENV ENTRYPOINT=/root/entrypoint.sh
RUN echo "#! /usr/bin/env bash" >> $ENTRYPOINT && \
    echo "source $SPACK_ROOT/share/spack/setup-env.sh" >> $ENTRYPOINT && \
    echo "unset CURRENTLY_BUILDING_DOCKER_IMAGE" >> $ENTRYPOINT && \
    echo "spack env activate -d $NESO_ROOT" >> $ENTRYPOINT && \
    echo "spack load mpich" >> $ENTRYPOINT && \
    echo "export PATH=$PATH:$NESO_ROOT/views/gcc-hipsycl/bin/" >> $ENTRYPOINT && \
    echo "exec \"\$@\"" >> $ENTRYPOINT && \
    chmod +x $ENTRYPOINT
ENTRYPOINT ["/bin/bash", "-c", "exec $ENTRYPOINT \"${@}\"", "--"]
