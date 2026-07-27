"""M3D-Modernized public package metadata.

Heavy model modules are intentionally not imported here.  Keeping package
initialisation lightweight allows data preparation and release-audit commands
to run without loading Transformers, MONAI or CUDA.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
