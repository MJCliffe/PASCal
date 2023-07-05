"""This module defines some integration tests
on common inputs to the app.

"""

from pathlib import Path

from PASCal.app import app

from pytest import fixture
from bs4 import BeautifulSoup


@fixture(scope="session")
def client():
    with app.test_client() as cli:
        yield cli


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


def test_T_sample_data(client, sample_T):
    post_parameters = {
        "DataType": "Temperature",
        "PcVal": "",
        "DegPolyCap": "",
        "DegPolyVol": "",
        "EulerianStrain": "True",
        "FiniteStrain": "True",
        "data": sample_T,
    }

    response = client.post("/output", data=post_parameters)
    assert response.status_code == 200

    html_response = [d for d in response.response]
    assert len(html_response) == 1

    soup = BeautifulSoup(html_response[0])
    tables = soup.find_all("table")
    assert len(tables) == 5


def test_P_sample_data(client, sample_P):
    post_parameters = {
        "DataType": "Pressure",
        "EulerianStrain": "True",
        "FiniteStrain": "True",
        "DegPolyCap": "",
        "DegPolyVol": "",
        "Pc": "True",
        "PcVal": 0.19,
        "data": sample_P,
    }

    response = client.post("/output", data=post_parameters)
    assert response.status_code == 200

    html_response = [d for d in response.response]
    assert len(html_response) == 1

    soup = BeautifulSoup(html_response[0])
    tables = soup.find_all("table")
    assert len(tables) == 7


def test_q_sample_data(client, sample_q):
    post_parameters = {
        "DataType": "Electrochemical",
        "EulerianStrain": "True",
        "FiniteStrain": "True",
        "Pc": "False",
        "PcVal": "",
        "DegPolyCap": 4,
        "DegPolyVol": 5,
        "data": sample_q,
    }

    response = client.post("/output", data=post_parameters)
    assert response.status_code == 200

    html_response = [d for d in response.response]
    assert len(html_response) == 1

    soup = BeautifulSoup(html_response[0])
    tables = soup.find_all("table")
    assert len(tables) == 6
