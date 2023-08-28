import importlib.metadata
from PASCal.core import fit, PASCalResults

__version__ = importlib.metadata.version("PASCal")

__all__ = ("__version__", "fit", "PASCalResults")
