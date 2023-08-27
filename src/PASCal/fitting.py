from typing import List, Callable
import numpy as np
import statsmodels.api as sm


def fit_linear_wls(
    y: np.ndarray, x: np.ndarray, x_errors: np.ndarray
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
    n_models = y.shape[1] if len(y.shape) > 1 else 1
    for i in range(n_models):
        X = sm.add_constant(x)
        model = sm.WLS(y[:, i], X, weights=1 / x_errors)

        fit_results.append(model.fit())

    return fit_results


def fit_empirical(
    strain,
    pressure,
    pressure_errors,
    empirical_function: Callable,
    linear_fit_results: List[sm.regression.linear_model.RegressionResultsWrapper],
):
    from scipy import curve_fit

    bounds = np.array(
        [
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.min(pressure), np.inf],
        ]
    )

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
        strain,
        bounds=bounds,
        sigma=pressure_errors,
        maxfev=5000,
    )
