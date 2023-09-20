import importlib.metadata

from PASCal.core import PASCalResults, fit

__version__ = importlib.metadata.version("PASCal")

__all__ = ("__version__", "fit", "PASCalResults")
