from typing import List, Callable, Tuple, Union, Dict, Optional
from functools import partial

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias
import numpy as np
import statsmodels.api as sm

from scipy.optimize import curve_fit
from PASCal.utils import (
    empirical_pressure_strain_relation,
)


Strain: TypeAlias = np.ndarray
Pressure: TypeAlias = np.ndarray
Volume: TypeAlias = np.ndarray
Charge: TypeAlias = np.ndarray
Temperature: TypeAlias = np.ndarray


def fit_linear_wls(
    y: Union[Strain, Volume],
    x: Union[Pressure, Temperature, Charge],
    x_errors: Union[Pressure, Temperature, Charge],
) -> List[sm.regression.linear_model.RegressionResultsWrapper]:
    """Fit a linear WLS models to the columns of the passed y array.

    Parameters;
        y: An array of dependent variables as a function of x, e.g., strain, volume.
        x: The control variable, e.g., pressure, temperature.
        x_errors: The errors in the control variable to use as weights for fit

    Returns:
        The results of the fit.

    """
    fit_results = []
    if len(y.shape) > 1:
        n_models = y.shape[1]
        _y = y
    else:
        n_models = 1
        _y = y.reshape(-1, 1)

    for i in range(n_models):
        X = sm.add_constant(x)
        model = sm.WLS(_y[:, i], X, weights=1 / x_errors)

        fit_results.append(model.fit())

    return fit_results


def fit_chebyshev(
    y: Union[Strain, Volume],
    x: Charge,
    max_degree: int,
):
    """Fit a Chebyshev polynomial to the passed y array of
    strains or volumes.

    Parameters:
        y: An array of dependent variables as a function of x, e.g., strain, volume.
        x: The control variable, e.g., pressure, temperature.
        max_degree: The maximum degree of Chebyshev polynomial to fit.

    Returns:
        The results of the fit.

    """
    coeffs, residuals, _, _, _ = np.polynomial.chebyshev.chebfit(
        x, y, max_degree, full=True
    )
    return coeffs, residuals


def fit_empirical_strain_pressure(
    diagonal_strain: Strain,
    pressure: Pressure,
    pressure_errors: Pressure,
    linear_fit_results: List[sm.regression.linear_model.RegressionResultsWrapper],
    empirical_function: Callable[
        [Pressure, Strain, Tuple[float, ...]], np.ndarray
    ] = empirical_pressure_strain_relation,
) -> Tuple[List[np.ndarray], List[np.ndarray], Callable]:
    """Fit an empirical function to the diagonal strain vs pressure data.

    Parameters:
        diagonal_strain: The array of diagonal strains.
        pressure: The array of pressures.
        pressure_errors: The errors in the pressure.
        linear_fit_results: A list of statsmodel linear regression results
            for each direction, used as a preconditioner.
        empirical_function: The function to fit to the data which accepts
            pressure and `popt` fit parameters as arguments.
            (default: `PASCal.utils.empirical_pressure`),

    Returns:
        A tuple of `popt` and `pcov` results for each strain direction,
        and the empirical function used.

    """

    bounds = np.array(
        [
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.min(pressure), np.inf],
        ]
    )

    popts: List[np.ndarray] = []
    pcovs: List[np.ndarray] = []

    for i in range(3):
        # todo check param order
        init_params = np.array(
            [
                linear_fit_results[i].params[1],
                linear_fit_results[i].params[0],
                np.min(pressure) - 0.001,
                0.5,
            ]
        )

        popt, pcov = curve_fit(
            empirical_function,
            pressure,
            diagonal_strain[:, i],
            p0=init_params,
            bounds=bounds,
            sigma=pressure_errors,
            maxfev=5000,
        )

        popts.append(popt)
        pcovs.append(pcov)

    return popts, pcovs, empirical_function


def fit_birch_murnaghan_volume_pressure(
    cell_volumes: Volume,
    pressure: Pressure,
    pressure_errors: Pressure,
    critical_pressure: Optional[float] = None,
) -> Tuple[Dict[Callable, np.ndarray]]:
    """Make a series of Birch-Murnaghan (BM) fits to the cell volume vs pressure data.

    Parameters:
        cell_volumes: The array of cell volumes.
        pressure: The array of pressures.
        pressure_errors: The errors in the pressure.
        critical_pressure: A critical pressure to use for the modified 3rd order BM fit (optional).

    Returns:
        Two dictionaries of popt and pcov results, keyed by the fitting function used.

    """
    from PASCal.utils import (
        birch_murnaghan_2nd,
        birch_murnaghan_3rd,
        birch_murnaghan_3rd_pc,
    )

    popts: Dict[Callable, np.ndarray] = []
    pcovs: Dict[Callable, np.ndarray] = []

    dp = pressure[-1] - pressure[0]
    dV = cell_volumes[-1] - cell_volumes[0]
    b_prime = -cell_volumes[0] * dp / dV

    # init params will use the last fit each round, so start with 2nd order
    init_params_2nd: np.ndarray = np.array(
        [
            cell_volumes[0],
            b_prime,
        ]
    )

    popts[birch_murnaghan_2nd], pcovs[birch_murnaghan_2nd] = curve_fit(
        birch_murnaghan_2nd,
        cell_volumes,
        pressure,
        p0=init_params_2nd,
        sigma=pressure_errors,
        maxfev=5000,
    )

    init_params_3rd: np.ndarray = (
        np.array([cell_volumes[0], popts[birch_murnaghan_2nd][1], b_prime / dp]),
    )

    popts[birch_murnaghan_3rd], pcovs[birch_murnaghan_3rd] = curve_fit(
        birch_murnaghan_3rd,
        cell_volumes,
        pressure,
        p0=init_params_3rd,
        sigma=pressure_errors,
        maxfev=5000,
    )

    if critical_pressure is not None:
        popts[birch_murnaghan_3rd_pc], pcovs[birch_murnaghan_3rd_pc] = curve_fit(
            partial(birch_murnaghan_3rd_pc, p_c=critical_pressure),
            cell_volumes,
            pressure,
            p0=popts[birch_murnaghan_3rd],
            sigma=pressure_errors,
            maxfev=5000,
        )

    return popts, pcovs
