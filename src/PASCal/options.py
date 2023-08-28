"""This module defines the options used by the PASCal library/app
when fitting strain and volume data.
"""
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np


class PASCalDataType(Enum):
    PRESSURE = "Pressure"
    ELECTROCHEMICAL = "Electrochemical"
    TEMPERATURE = "Temperature"


@dataclass
class Options:
    data_type: PASCalDataType = field(
        metadata={
            "form": "DataType",
        },
    )
    """The type of data passed, either 'Pressure', 'Electrochemical' or 'Temperature'."""

    eulerian_strain: bool = field(
        default=True,
        metadata={
            "form": "EulerianStrain",
        },
    )
    """Whether to use Eulerian strain (True) or Lagrangian"""

    finite_strain: bool = field(default=True, metadata={"form": "FiniteStrain"})
    """Whether to use finite strain (True) or infinitesimal strain"""

    use_pc: bool = field(
        default=False,
        metadata={
            "form": "Pc",
        },
    )
    """Whether to use the critical pressure to modify the fits."""

    pc_val: Optional[float] = field(
        default=None,
        metadata={
            "form": "PcVal",
        },
    )
    """The critical pressure value to use in GPa."""

    deg_poly_strain: int = field(
        default=5,
        metadata={
            "form": "DegPolyCap",
        },
    )
    """The degree of polynomial to use for fitting the strain vs charge/capacity."""

    deg_poly_vol: int = field(
        default=5,
        metadata={
            "form": "DegPolyVol",
        },
    )
    """The degree of polynomial to use for fitting the volume vs charge/capacity"""

    @staticmethod
    def from_dict(options: Dict[str, Any]) -> "Options":
        """Load options from a dictionary."""

        if options.get("data_type"):
            options["data_type"] = PASCalDataType[options["data_type"].upper()]
        if options.get("pc_val"):
            options["pc_val"] = float(options["pc_val"])
        if options.get("deg_poly_strain"):
            options["deg_poly_strain"] = int(options["deg_poly_strain"])
        if options.get("deg_poly_vol"):
            options["deg_poly_vol"] = int(options["deg_poly_vol"])

        return Options(**options)

    @staticmethod
    def from_form(form_data: Dict[str, str]) -> "Options":
        """Go through and check the values provided by the form."""

        options = {}

        for key in fields(Options):
            form_key = key.metadata["form"]
            if form_key in form_data:
                value = True if form_data[form_key] == "True" else form_data[form_key]
                value = False if form_data[form_key] == "False" else value
                options[key.name] = value

        return Options.from_dict(options)

    def precheck_inputs(self, x) -> List[str]:
        """Check that the raw data passed is compatible with the options, adjusting
        the options where possible.

        Returns:
            The adjusted options and a list of warnings.

        """
        if len(x) < 2:
            raise RuntimeError("Too few data points to perform fit: need at least 2")

        warning: List[str] = []

        if self.data_type == PASCalDataType.PRESSURE:
            if len(x) < 4:
                warning.append(
                    "At least as many data points as parameters are needed for a fit to be carried out (e.g. 3 for 3rd order Birch-Murnaghan, 4 for empirical pressure fitting). "
                    "As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."
                )
            if self.use_pc and self.pc_val:
                if np.amin(x) < self.pc_val:
                    pc_val = np.min(x)
                    warning.append(
                        "The critical pressure has to be smaller than the lower pressure data point. "
                        f"Critical pressure has been set to the minimum value: {pc_val} GPa."
                    )
                    self.pc_val = pc_val

        if self.data_type == PASCalDataType.ELECTROCHEMICAL:
            if len(x) - 2 < self.deg_poly_strain:
                deg_poly_strain = len(x) - 2
                warning.append(
                    f"The maximum degree of the Chebyshev strain polynomial has been lowered from {options.deg_poly_strain} to {deg_poly_strain}. "
                    "At least as many data points as parameters are needed for a fit to be carried out. "
                    "As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."
                )
                self.deg_poly_strain = deg_poly_strain
            if len(x) - 2 < self.deg_poly_vol:
                deg_poly_vol = len(x) - 2
                warning.append(
                    f"The maximum degree of the Chebyshev volume polynomial has been lowered from {options.deg_poly_vol} to {deg_poly_vol}. "
                    "At least as many data points as parameters are needed for a fit to be carried out. "
                    "As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."
                )
                self.deg_poly_vol = deg_poly_vol

        return warning
