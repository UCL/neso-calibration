"""Wrappers for NESO models."""

import asyncio
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import parse as xml_parse

import aiofiles


class ProcessOutput(NamedTuple):
    """Output data from process."""

    return_code: int | None
    stdout: bytes
    stderr: bytes


ParametersDict = dict[str, float | int]
ExtractOutputsFunction = Callable[
    [Path, ProcessOutput, ParametersDict],
    Any,
]
PathLike = str | Path


async def _redirect_stream(input_stream, output_stream) -> bytes:
    """Asynchronously redirect lines from input_stream to output_stream.

    Also returns concatenated output.
    """
    output = b""
    while True:
        line = await input_stream.readline()
        if line:
            output += line
            output_stream.write(line.decode())
        else:
            break
    return output


class AbstractModel(ABC):
    """Base class for wrappers for executing a NESO solver and extracting outputs."""

    def __init__(
        self,
        solver_executable: PathLike,
        base_session_file_path: PathLike,
        mesh_file_path: PathLike,
        extract_outputs_function: ExtractOutputsFunction,
        *,
        environment_variables: dict[str, str] | None = None,
        setup_commands: Sequence[str] = (),
        num_omp_threads: int = 1,
        num_mpi_processes: int = 1,
        mpirun_options: str = "",
        redirect_subprocess_streams: bool = False,
    ):
        """Create a new NESO model wrapper instance.

        Args:
            solver_executable: NESO solver executable to use. If the solver executable
                is on the current search path then the solver name can be used directly,
                otherwise a path to the executable should be provided.
            base_session_file_path: Path to XML file defining bae NESO configuration.
                Parameters values defined in this file are used as defaults unless
                overridden by `parameter_overrides` keyword arguments when calling
                model.
            mesh_file_path: Path to XML file defining spatial mesh to solve on.
            extract_outputs_function: Function to extract required outputs from model.
                This function is passed the path to the temporary directory any outputs
                from solver executable are written to, a `ProcessOutput` tuple with
                return code and captured `stdout` and `stderr` outputs from subprocess
                used to execute the model run and a dictionary of all of the parameter
                values in the session file used to run the model. The return value of
                this function is returned when calling the model.

        Keyword Args:
            environment_variables: Any additional environment variables to set in
                environment used to execute solver.
            setup_commands: Any commands to execute in environment used to execute
                solver before executing solver command (for example to setup
                environment), as a sequence of strings.
            num_omp_threads: Value to set OMP_NUM_THREADS environment variable
                specifying number of OpenMP threads to use, in local environment that
                solver is executed.
            num_mpi_processes: Number of message passing interface (MPI) processes to
                run solver executable with. If set to 1 (the default) solver exectuable
                is run directly if set to greater than 1 then solver exectuable
                command is passed to `mpirun` with `-n` argument set to specified number
                of processes.
            mpirun_options: Any additional optional arguments to pass to `mpirun`
                command (only used when `num_mpi_processes > 1`).
            redirect_subprocess_streams: Whether to redirect the `stdout` and `stderr`
                output streams of the subprocess used to run the model in to the
                corresponding output streams of the output process. Enabling this
                redirection can be useful for getting live output from the model while
                it is running. If set to `False` the completed process object passed to
                the `extract_outputs` function has string `stdout` and `stderr`
                attributes which can be used to access any text written to the output
                streams after the model run subprocess has completed.
        """
        self._solver_executable = solver_executable
        for path in (mesh_file_path, base_session_file_path):
            if not Path(path).exists():
                msg = f"No file exists at {path}"
                raise FileNotFoundError(msg)
        self._mesh_file_path = Path(mesh_file_path).resolve()
        self._base_session_file_path = Path(base_session_file_path).resolve()
        self._extract_outputs_function = extract_outputs_function
        self._environment_variables = (
            {} if environment_variables is None else environment_variables
        )
        self._setup_commands = tuple(setup_commands)
        self._environment_variables["NUM_OMP_THREADS"] = str(num_omp_threads)
        self._num_mpi_processes = num_mpi_processes
        self._mpirun_options = mpirun_options
        self._redirect_subprocess_streams = redirect_subprocess_streams

    def _write_session_file_and_read_parameters(
        self,
        output_path: Path,
        **parameter_overrides: float | int,
    ) -> ParametersDict:
        tree = xml_parse(self._base_session_file_path)  # noqa: S314
        root = tree.getroot()
        parameters_container = root.find("CONDITIONS").find(  # type: ignore[union-attr]
            "PARAMETERS",
        )  # type: ignore[union-attr]
        default_parameter_values = {}
        parameter_elements_to_remove = []
        for element in parameters_container.iter("P"):  # type: ignore[union-attr]
            match = re.match(
                r"\s*(?P<key>\w+)\s*=\s*(?P<value>[0-9.-]+)\s*",
                element.text,  # type: ignore[arg-type]
            )
            if match is None:
                msg = (
                    f"Invalid parameter specification {element.text} "
                    "in base session file"
                )
                raise ValueError(msg)
            key = match.group("key")
            value = match.group("value")
            default_parameter_values[key] = float(value) if "." in value else int(value)
            if key in parameter_overrides:
                parameter_elements_to_remove.append(element)
        # Remove parameter elements that will be added when setting overrides.
        # We do this outside above loop to avoid changing the container while iterating
        # over it.
        for parameter_element in parameter_elements_to_remove:
            parameters_container.remove(parameter_element)  # type: ignore[union-attr]
        for key, value in parameter_overrides.items():
            parameter_element = Element("P")
            parameter_element.text = f"{key} = {value}"
            parameters_container.append(parameter_element)  # type: ignore[union-attr]
        tree.write(output_path)
        return default_parameter_values | parameter_overrides

    def _construct_run_command(
        self,
        session_file_path: PathLike,
        mesh_file_path: PathLike,
    ) -> str:
        base_command = f"{self._solver_executable} {session_file_path} {mesh_file_path}"
        if self._num_mpi_processes > 1:
            mpi_command = f"mpirun -n {self._num_mpi_processes} {self._mpirun_options}"
            return f"{mpi_command} {base_command}"
        else:
            return base_command

    async def _create_subprocess(
        self,
        cmd: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ProcessOutput:
        process = await asyncio.create_subprocess_shell(
            cmd=cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        if self._redirect_subprocess_streams:
            stdout, stderr = await asyncio.gather(
                _redirect_stream(process.stdout, sys.stdout),
                _redirect_stream(process.stderr, sys.stderr),
            )
            await process.wait()
        else:
            stdout, stderr = await process.communicate()
        return ProcessOutput(process.returncode, stdout, stderr)

    @abstractmethod
    async def _run_model_subprocess(
        self,
        session_file_path: Path,
        temporary_directory_path: Path,
    ) -> ProcessOutput:
        """Run model with specified session file outputting to temporary directory.

        Args:
            session_file_path: Path to session file to run solver with.
            temporary_directory: Path to directory to which model outputs should be
                written.

        Returns:
            Tuple of return code and captured output of subprocess used for model run.
        """

    async def __call__(self, **parameter_overrides: float | int):
        """Asynchronously run model with specified parameter overrides.

        Keyword Args:
            parameter_overrides: Values to override default parameters in session file
               used to run model with.

        Returns:
            Outputs returned by `extract_outputs` function specified when constructing
            model instance.
        """
        async with aiofiles.tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory_path = Path(temporary_directory)
            session_file_path = temporary_directory_path / "session.xml"
            parameter_values = self._write_session_file_and_read_parameters(
                session_file_path,
                **parameter_overrides,
            )
            process_output = await self._run_model_subprocess(
                session_file_path,
                temporary_directory_path,
            )
            return self._extract_outputs_function(
                temporary_directory_path,
                process_output,
                parameter_values,
            )


class NativeModel(AbstractModel):
    """NESO model using solver installed natively on local filesystem."""

    async def _run_model_subprocess(
        self,
        session_file_path: Path,
        temporary_directory_path: Path,
    ) -> ProcessOutput:
        run_command = self._construct_run_command(
            session_file_path,
            self._mesh_file_path,
        )
        return await self._create_subprocess(
            " && ".join((*self._setup_commands, run_command)),
            cwd=str(temporary_directory_path),
            env=os.environ | self._environment_variables,
        )


class DockerModel(AbstractModel):
    """NESO model using solver installed in Docker image."""

    def __init__(
        self,
        image_name: str,
        solver_executable: PathLike,
        base_session_file_path: PathLike,
        mesh_file_path: PathLike,
        extract_outputs_function: ExtractOutputsFunction,
        *,
        setup_commands: Sequence[str] = (),
        environment_variables: dict[str, str] | None = None,
        container_mount_path: PathLike = "/output",
        container_entry_point: str | None = None,
        num_omp_threads: int = 1,
        num_mpi_processes: int = 1,
        mpirun_options: str = "",
        redirect_subprocess_streams: bool = False,
    ):
        """Create a new Docker based NESO model wrapper instance.

        Args:
            image_name: Name of Docker image in which NESO solvers are installed.
            solver_executable: NESO solver executable to use. If the solver executable
                is on the current search path (on the Docker image) then the solver name
                can be used directly, otherwise a path to the executable on the
                container file system should be provided.
            base_session_file_path: Path to XML file defining base NESO configuration,
                on local file system. Parameters values defined in this file are used as
                defaults unless overridden by `parameter_overrides` keyword arguments
                when calling model.
            mesh_file_path: Path to XML file defining spatial mesh to solve on, on local
                file system.
            extract_outputs_function: Function to extract required outputs from model.
                This function is passed the path to the temporary directory any outputs
                from solver executable are written to, a `ProcessOutput` tuple with
                return code and captured `stdout` and `stderr` outputs from subprocess
                used to execute the model run and a dictionary of all of the parameter
                values in the session file used to run the model. The return value of
                this function is returned when calling the model.

        Keyword Args:
            setup_commands: Any commands to execute in Docker container shell before
                executing solver command (for example to setup environment within
                container), as a sequence of strings.
            environment_variables: Any additional environment variables to set in
                container used to execute solver.
            container_mount_path: Path to bind mount on container the temporary
                directory on local host system used for writing outputs to and as
                working directory for solver executions. No directory or file should
                already exist at this path on the container.
            container_entrypoint: Entrypoint for container, specifying command executed
                when container is started and to which the run arguments are passed. If
                :code:`None` (the default) then the default entrypoint for the image
                will be used, otherwise the command specified by this argument will be
                used.
            num_omp_threads: Value to set OMP_NUM_THREADS environment variable
                specifying number of OpenMP threads to use, in local environment that
                solver is executed.
            num_mpi_processes: Number of message passing interface (MPI) processes to
                run solver executable with. If set to 1 (the default) solver exectuable
                is run directly if set to greater than 1 then solver exectuable
                command is passed to `mpirun` with `-n` argument set to specified number
                of processes.
            mpi_run_options: Any additional optional arguments to pass to `mpirun`
                command (only used when `num_mpi_processes > 1`).
            redirect_subprocess_streams: Whether to redirect the `stdout` and `stderr`
                output streams of the subprocess used to run the model in to the
                corresponding output streams of the output process. Enabling this
                redirection can be useful for getting live output from the model while
                it is running. If set to `False` the completed process object passed to
                the `extract_outputs` function has string `stdout` and `stderr`
                attributes which can be used to access any text written to the output
                streams after the model run subprocess has completed.
        """
        super().__init__(
            solver_executable=solver_executable,
            base_session_file_path=base_session_file_path,
            mesh_file_path=mesh_file_path,
            extract_outputs_function=extract_outputs_function,
            setup_commands=setup_commands,
            environment_variables=environment_variables,
            num_omp_threads=num_omp_threads,
            num_mpi_processes=num_mpi_processes,
            mpirun_options=mpirun_options,
            redirect_subprocess_streams=redirect_subprocess_streams,
        )
        self._image_name = image_name
        self._container_mount_path = container_mount_path
        self._container_entry_point = container_entry_point

    async def _run_model_subprocess(
        self,
        session_file_path: Path,
        temporary_directory_path: Path,
    ) -> ProcessOutput:
        # Ensure copies of session and mesh files in temporary directory that container
        # will have access to.
        if session_file_path.parent != temporary_directory_path:
            shutil.copy(
                session_file_path,
                temporary_directory_path / session_file_path.name,
            )
        mesh_file_path = temporary_directory_path / self._mesh_file_path.name
        shutil.copy(self._mesh_file_path, mesh_file_path)
        # Get paths to session and mesh files within _container file system_.
        container_mount_path = Path(self._container_mount_path)
        container_session_file_path = container_mount_path / session_file_path.name
        container_mesh_file_path = container_mount_path / mesh_file_path.name
        run_command = self._construct_run_command(
            container_session_file_path,
            container_mesh_file_path,
        )
        container_commands = [
            *self._setup_commands,
            *[f"export {k}={v}" for k, v in self._environment_variables.items()],
            run_command,
            # We need to change user and group owner for any files written by container
            # to match current user and group IDs to avoid file permission errors when
            # cleaning up temporary directory.
            f"chown -R {os.getuid()}:{os.getgid()} {container_mount_path}",
        ]
        entrypoint_argument = (
            ""
            if self._container_entry_point is None
            else f"--entrypoint {self._container_entry_point}"
        )
        docker_command = (
            f"docker run --rm "
            f"-v {temporary_directory_path}:{container_mount_path}:rw  "
            f"-w {container_mount_path} {entrypoint_argument}"
            f"{self._image_name} /bin/bash -c '{' && '.join(container_commands)}'"
        )
        return await self._create_subprocess(
            docker_command,
            cwd=str(temporary_directory_path),
        )
