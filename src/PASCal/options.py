from dataclasses import dataclass, field, fields
from typing import Optional, Dict
from enum import Enum

class PASCalDataType(Enum):
    PRESSURE = "Pressure"
    ELECTROCHEMICAL = "Electrochemical"
    TEMPERATURE = "Temperature"


@dataclass
class Options:
    eulerian_strain: bool = field(
        default=True,
        metadata={
            "form": "EulerianStrain",
            "description": "Whether to use Eulerian strain (True) or Lagrangian",
        },
    )
    finite_strain: bool = field(default=True, metadata={"form": "FiniteStrain"})
    data_type: PASCalDataType = field(
        default=None,
        metadata={
            "form": "DataType",
            "description": "The type of data passed, either 'Pressure', 'Electrochemical' or 'Temperature'",
        },
    )
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
    deg_poly_cap: Optional[int] = field(
        default=None,
        metadata={
            "form": "DegPolyCap",
            "description": "The degree of polynomial to use for fitting the capacity.",
        },
    )
    deg_poly_vol: Optional[int] = field(
        default=None,
        metadata={
            "form": "DegPolyVol",
            "description": "The degree of polynomial to use for fitting the volume.",
        },
    )

    @staticmethod
    def from_form(self, form_data: Dict[str, str]) -> Options:
    """Go through the set values of options and check their values."""

    options = {}

    for key in fields(Options):
        form_key = key.metadata["form"]
        if form_key in form_data:
            value = True if form_data[form_key] == "True" else form_data[form_key]
            value = False if form_data[form_key] == "False" else value
            options[key.name] = value

    if options.get("data_type"):
        options["data_type"] = PASCalDataType(options["data_type"])
    if options.get("pc_val"):
        options["pc_val"] = float(options["pc_val"])
    if options.get("deg_poly_cap"):
        options["deg_poly_cap"] = int(options["deg_poly_cap"])
    if options.get("deg_poly_vol"):
        options["deg_poly_vol"] = int(options["deg_poly_vol"])

    return Options(**options)
