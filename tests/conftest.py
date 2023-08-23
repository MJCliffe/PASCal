from pytest import fixture
from pathlib import Path

def load_sample_data(filename):
    with open(
        Path(__file__).parent / "data" / filename, "r", encoding="utf-8"
    ) as f:
        flines = f.readlines()
    return "\r".join(flines).replace("\n\r", "\r\n").strip("\n")


@fixture
def sample_T():
    return load_sample_data("sample_T.tsv")


@fixture
def sample_P():
    return load_sample_data("sample_P.tsv")


@fixture
def sample_q():
    return load_sample_data("sample_q.tsv")
