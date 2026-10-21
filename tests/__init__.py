"""Import torch before numpy/pandas on this Windows environment.

Loading torch after numpy 1.26.4 / pandas 2.0.3 here fails with
"WinError 1114 ... c10.dll"; importing it first avoids that DLL clash and
leaves the lazy ``import torch`` inside the MLP code unchanged.
"""

try:  # pragma: no cover - environment-dependent import ordering
    import torch as _torch_preload  # noqa: F401
except Exception:  # torch is optional for every non-MLP configuration
    _torch_preload = None
