"""kt_epimodel_hira — Sudogwon metapop influenza model (HIRA calibration target).

Sister project of kt_epimodel; ILI target replaced with HIRA episode counts.
See README.md and docs/SKELETON_ANALYSIS.md (section 6) for design.
"""

__version__ = "0.1.0"

import os as _os

_os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{_os.path.expanduser('~/Documents/python/NIMS/kt_epimodel_hira/outputs/mlruns/mlflow.db')}",
)
_os.environ.setdefault(
    "MLFLOW_DEFAULT_ARTIFACT_ROOT",
    _os.path.expanduser("~/Documents/python/NIMS/kt_epimodel_hira/outputs/mlruns/artifacts"),
)
