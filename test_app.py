"""This module defines some integration tests
on common inputs to the app.

"""

from pathlib import Path
from pickle import load

from app import app

from pytest import fixture

@fixture(scope="session")
def client():
    with app.test_client() as cli:
        yield cli

def load_sample_data(filename):
    with open(Path(__file__).parent / "tests" / "data" / filename, "r", encoding="utf-8") as f:
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
        "finite": 'True',
        "data": sample_T, 
    }

    response = client.post("/output", data=post_parameters)
    assert response.status_code == 200

    html_response = [d for d in response.response]
    assert len(html_response) == 1

    # Do some beautiful soup HTML parsing here, check for:
    #   i) correct values in data tables and 
    #   ii) existence of plots


def test_P_sample_data(client, sample_P):

    post_parameters = {
        "DataType": "Pressure",
        "EulerianStrain": "True",
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

    # Do some beautiful soup HTML parsing here, check for:
    #   i) correct values in data tables and 
    #   ii) existence of plots


def test_q_sample_data(client, sample_q):
    post_parameters = {
        "DataType": "Electrochemical",
        "EulerianStrain": "True",
        "finite": "True",
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

    # Do some beautiful soup HTML parsing here, check for:
    #   i) correct values in data tables and 
    #   ii) existence of plots
