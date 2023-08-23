"""This module defines some integration tests
on common inputs to the app.

"""

from functools import partial

from PASCal.app import app

from pytest import fixture
from bs4 import BeautifulSoup


@fixture(scope="session")
def client():
    with app.test_client() as cli:
        yield cli

@fixture(scope="session")
def parser():
    return partial(BeautifulSoup, features="html.parser")

def test_T_sample_data(client, sample_T, parser):
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

    soup = parser(html_response[0])
    tables = soup.find_all("table")
    assert len(tables) == 5


def test_P_sample_data(client, sample_P, parser):
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

    soup = parser(html_response[0])
    tables = soup.find_all("table")
    assert len(tables) == 7


def test_q_sample_data(client, sample_q, parser):
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

    soup = parser(html_response[0])
    tables = soup.find_all("table")
    assert len(tables) == 6
