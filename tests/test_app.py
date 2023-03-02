"""This module defines some integration tests
on common inputs to the app.

"""

import app
from pathlib import Path
from pytest import fixture

@fixture(scope="session")
def client():
    with app.test_client() as cli:
        yield cli

def test_T_sample_data(client):
    with open(Path(__file__).parent / "data" / "T_sample.tsv", "r") as f:
        flines = f.readlines()

    post_parameters = {
        "data": "\n".join(flines),
        "temp": True,
        "eulerian": True,
        "finite": True,
        "Pc": False,
        "PcVal": "",
        "DegPolyCap": "",
        "DegPolyVol": "",
    }

    client.post(post_parameters)


def test_P_sample_data():
    pass

def test_q_sample_data():
    pass
