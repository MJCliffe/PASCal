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


def check_tables(table, expected_table):
    """Check tables but allow rounding from 0.0 to -0.0."""
    for ind, _ in enumerate(table):
        if expected_table[ind] == "-0.0":
            expected_table[ind] = "0.0"
        if table[ind] == "-0.0":
            table[ind] = "0.0"

    assert table == expected_table
    return True


def test_T_sample_data(
    client,
    sample_T,
    parser,
    sample_T_output,
    sample_T_delta_length,
    sample_T_volumes,
    sample_T_principal_axes,
):
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

    assert check_tables(
        [td.text for td in tables[0].find_all("td")], sample_T_output
    ), "Output table failed"

    assert check_tables(
        [td.text for td in tables[1].find_all("td")], sample_T_delta_length
    ), "% length table failed"

    assert check_tables(
        [td.text for td in tables[2].find_all("td")], sample_T_principal_axes
    ), "Principal axes table failed"

    assert check_tables(
        [td.text for td in tables[3].find_all("td")], sample_T_volumes
    ), "Volume table failed"


def test_p_sample_data(
    client,
    sample_p,
    parser,
    sample_p_output,
    sample_p_bm,
    sample_p_K,
    sample_p_delta_length,
    sample_p_principal_axes,
    sample_p_volumes,
):
    post_parameters = {
        "DataType": "Pressure",
        "EulerianStrain": "True",
        "FiniteStrain": "True",
        "DegPolyCap": "",
        "DegPolyVol": "",
        "UsePc": "True",
        "PcVal": "0.19",
        "data": sample_p,
    }

    response = client.post("/output", data=post_parameters)
    assert response.status_code == 200

    html_response = [d for d in response.response]
    assert len(html_response) == 1

    soup = parser(html_response[0])
    tables = soup.find_all("table")
    assert len(tables) == 7

    assert check_tables(
        [td.text for td in tables[0].find_all("td")], sample_p_output
    ), "Output table failed"

    assert check_tables(
        [td.text for td in tables[1].find_all("td")], sample_p_bm
    ), "BM table failed"

    assert check_tables(
        [td.text for td in tables[2].find_all("td")], sample_p_K
    ), "Compressibility table failed"

    assert check_tables(
        [td.text for td in tables[3].find_all("td")], sample_p_delta_length
    ), "% length table failed"

    assert check_tables(
        [td.text for td in tables[4].find_all("td")], sample_p_principal_axes
    ), "Principal axes table failed"

    assert check_tables(
        [td.text for td in tables[5].find_all("td")], sample_p_volumes
    ), "Volume table failed"


def test_q_sample_data(
    client,
    sample_q,
    parser,
    sample_q_output,
    sample_q_volumes,
    sample_q_principal_axes,
    sample_q_charge_derivative,
    sample_q_delta_length,
):
    post_parameters = {
        "DataType": "Electrochemical",
        "EulerianStrain": "True",
        "FiniteStrain": "True",
        "UsePc": "False",
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

    check_tables(
        [td.text for td in tables[0].find_all("td")], sample_q_output
    ), "Output table failed"

    check_tables(
        [td.text for td in tables[1].find_all("td")], sample_q_delta_length
    ), "% length failed"

    check_tables(
        [td.text for td in tables[2].find_all("td")], sample_q_principal_axes
    ), "Principal axes table failed"

    check_tables(
        [td.text for td in tables[3].find_all("td")], sample_q_charge_derivative
    ), "Charge derivative table failed"

    check_tables(
        [td.text for td in tables[4].find_all("td")], sample_q_volumes
    ), "Volumes table failed"
