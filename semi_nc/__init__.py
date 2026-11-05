"""Semi-NC paper reproduction."""

from .config import load_config
from .model import SemiNCNet

__all__ = ["SemiNCNet", "load_config"]
__version__ = "0.1.0"
