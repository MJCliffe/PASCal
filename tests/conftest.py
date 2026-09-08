from pathlib import Path

from pytest import fixture


def load_sample_inputs(filename):
    with open(Path(__file__).parent / "data" / filename, "r", encoding="utf-8") as f:
        flines = f.readlines()
    return "\r".join(flines).replace("\n\r", "\r\n").strip("\n")


def load_sample_outputs(filename):
    with open(Path(__file__).parent / "data" / filename, "r", encoding="utf-8") as f:
        flines = f.readlines()
    return [d.strip() for line in flines for d in line.split("\t")]


@fixture
def sample_T():
    return load_sample_inputs("sample_T/input.tsv")


@fixture
def sample_p():
    return load_sample_inputs("sample_p/input.tsv")


@fixture
def sample_p_tricky():
    return load_sample_inputs("sample_p_tricky/input.tsv")


@fixture
def sample_q():
    return load_sample_inputs("sample_q/input.tsv")


@fixture
def sample_q_output():
    return load_sample_outputs("sample_q/output.tsv")


@fixture
def sample_q_volumes():
    return load_sample_outputs("sample_q/volumes.tsv")


@fixture
def sample_q_charge_derivative():
    return load_sample_outputs("sample_q/charge_derivative_strain.tsv")


@fixture
def sample_q_principal_axes():
    return load_sample_outputs("sample_q/principal_axes.tsv")


@fixture
def sample_q_delta_length():
    return load_sample_outputs("sample_q/change_in_length.tsv")


@fixture
def sample_p_output():
    return load_sample_outputs("sample_p/output.tsv")


@fixture
def sample_p_bm():
    return load_sample_outputs("sample_p/bm.tsv")


@fixture
def sample_p_K():
    return load_sample_outputs("sample_p/compressibilities.tsv")


@fixture
def sample_p_delta_length():
    return load_sample_outputs("sample_p/delta_length.tsv")


@fixture
def sample_p_principal_axes():
    return load_sample_outputs("sample_p/principal_axes.tsv")


@fixture
def sample_p_volumes():
    return load_sample_outputs("sample_p/volumes.tsv")


@fixture
def sample_T_output():
    return load_sample_outputs("sample_T/output.tsv")


@fixture
def sample_T_delta_length():
    return load_sample_outputs("sample_T/delta_length.tsv")


@fixture
def sample_T_principal_axes():
    return load_sample_outputs("sample_T/principal_axes.tsv")


@fixture
def sample_T_volumes():
    return load_sample_outputs("sample_T/volumes.tsv")
