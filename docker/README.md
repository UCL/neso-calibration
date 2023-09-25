This directory contains Dockerfiles which can be used to build Docker images in which NESO is built and installed using the GCC or Intel oneAPI compiler toolchains using Spack, and the environment set up to allow easily running the built NESO solvers including via MPI.

From this directory run

```sh
docker build -t neso-oneapi oneapi
```

to build an image which use the Intel oneAPI toolchain with tag `neso-oneapi` and run

```sh
docker build -t neso-gcc gcc
```

to build an image which use the GCC toolchain with tag `neso-gcc`.
