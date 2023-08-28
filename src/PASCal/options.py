from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any
from enum import Enum


class PASCalDataType(Enum):
    PRESSURE = "Pressure"
    ELECTROCHEMICAL = "Electrochemical"
    TEMPERATURE = "Temperature"


@dataclass
class Options:
    data_type: PASCalDataType = field(
        metadata={
            "form": "DataType",
            "description": "The type of data passed, either 'Pressure', 'Electrochemical' or 'Temperature'",
        },
    )
    eulerian_strain: bool = field(
        default=True,
        metadata={
            "form": "EulerianStrain",
            "description": "Whether to use Eulerian strain (True) or Lagrangian",
        },
    )
    finite_strain: bool = field(default=True, metadata={"form": "FiniteStrain"})
    use_pc: bool = field(
        default=False,
        metadata={
            "form": "Pc",
            "description": "Whether to use the critical pressure to modify the fits.",
        },
    )
    pc_val: Optional[float] = field(
        default=None,
        metadata={
            "form": "PcVal",
            "description": "Critical pressure value to use in GPa",
        },
    )
    deg_poly_strain: int = field(
        default=5,
        metadata={
            "form": "DegPolyCap",
            "description": "The degree of polynomial to use for fitting the strain vs charge/capacity.",
        },
    )
    deg_poly_vol: int = field(
        default=5,
        metadata={
            "form": "DegPolyVol",
            "description": "The degree of polynomial to use for fitting the volume vs charge/capacity",
        },
    )

    @staticmethod
    def from_dict(options: Dict[str, Any]) -> "Options":
        """Load options from a dictionary."""

        if options.get("data_type"):
            options["data_type"] = PASCalDataType[options["data_type"].upper()]
        if options.get("pc_val") is not None:
            options["pc_val"] = float(options["pc_val"])
        if options.get("deg_poly_strain") is not None:
            options["deg_poly_strain"] = int(options["deg_poly_strain"])
        if options.get("deg_poly_vol") is not None:
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
