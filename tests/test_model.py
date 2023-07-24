"""Tests for model module."""

import pytest

import nesopy.model

DUMMY_PARAMETERS_ONLY_SESSION_TEMPLATE = """
<NEKTAR xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://www.nektar.info/schema/nektar.xsd">
    <CONDITIONS>
        <PARAMETERS>
            {parameters}
        </PARAMETERS>
    </CONDITIONS>
</NEKTAR>
"""

DUMMY_MESH_FILE_CONTENTS = """
<NEKTAR>
    <GEOMETRY DIM="2" SPACE="2">
        <VERTEX>
            <V ID="0">0.00000000e+00 0.00000000e+00 0.00000000e+00</V>
            <V ID="1">5.00000000e-02 0.00000000e+00 0.00000000e+00</V>
            <V ID="2">5.00000000e-02 1.00000000e-02 0.00000000e+00</V>
            <V ID="3">0.00000000e+00 1.00000000e-02 0.00000000e+00</V>
        </VERTEX>
        <EDGE>
            <E ID="0">0 1</E>
            <E ID="1">1 2</E>
            <E ID="2">2 3</E>
            <E ID="3">3 0</E>
        </EDGE>
        <ELEMENT>
            <Q ID="0">0 1 2 3</Q>
        </ELEMENT>
        <CURVED />
        <COMPOSITE>
            <C ID="1"> Q[0] </C>
            <C ID="100"> E[0] </C>
            <C ID="200"> E[1] </C>
            <C ID="300"> E[2] </C>
            <C ID="400"> E[3] </C>
        </COMPOSITE>
        <DOMAIN>
            <D ID="0"> C[1] </D>
        </DOMAIN>
    </GEOMETRY>
</NEKTAR>
"""


def _dummy_session_file_contents(parameters: dict[str, float | int]):
    return DUMMY_PARAMETERS_ONLY_SESSION_TEMPLATE.format(
        parameters="\n".join([f"<P>{k}={v}</P>" for k, v in parameters.items()]),
    )


@pytest.mark.parametrize(
    "base_parameters",
    [{"test_0": 1}, {"test_1": 1.0, "test_2": 2}],
)
@pytest.mark.parametrize(
    "parameter_overrides",
    [{}, {"test_0": 2.0}, {"test_0": 0, "test_1": 2}],
)
def test_native_model(tmp_path, base_parameters, parameter_overrides):
    base_session_file_contents = _dummy_session_file_contents(base_parameters)
    base_session_file_path = tmp_path / "base_session.xml"
    with base_session_file_path.open("w") as f:
        f.write(base_session_file_contents)
    mesh_file_path = tmp_path / "mesh.xml"
    with mesh_file_path.open("w") as f:
        f.write(DUMMY_MESH_FILE_CONTENTS)
    executable_path = tmp_path / "DummySolver"
    with executable_path.open("w") as f:
        f.write("cat $1 $2")
    # Add executable permissions for user
    executable_path.chmod(0o700)

    def _dummy_extract_outputs(temporary_directory, completed_process, parameters):
        assert temporary_directory.exists()
        return completed_process, parameters

    model = nesopy.model.NativeModel(
        solver_executable_path=executable_path,
        base_session_file_path=base_session_file_path,
        mesh_file_path=mesh_file_path,
        extract_outputs_function=_dummy_extract_outputs,
    )

    completed_process, parameters = model(**parameter_overrides)
    assert parameters == (base_parameters | parameter_overrides)
    assert DUMMY_MESH_FILE_CONTENTS in completed_process.stdout.decode("utf-8")
