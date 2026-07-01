"""Shared pytest configuration.

Sets ``KMP_DUPLICATE_LIB_OK`` BEFORE any test imports torch/faiss. Both ship their own copy of
Intel's OpenMP runtime (libomp/libiomp5); when tests/test_recommend_v3.py imports ``api.app``
(which imports faiss and torch) in isolation, the double libomp initialization otherwise aborts
the process with a SIGABRT ("OMP: Error #15: Initializing libiomp5..."). Allowing the duplicate
load is the standard workaround for this dev/test setup.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
