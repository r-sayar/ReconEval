"""Reconstruction decoders.

``ReconMLPDecoder`` works with the standard ``cstm_scvi_env``.
``ReconTransformerDecoder`` depends on the ``concept`` package and
only runs under ``scconcept_env`` — its import is guarded so the
package stays importable elsewhere. Failures are logged to
``stderr`` (not silently swallowed) so RTD / CI build logs reveal
the root cause when a class goes missing.
"""

import sys as _sys

from ._base_decoder import BaseReconstructionDecoder

__all__ = ["BaseReconstructionDecoder"]


def _try_import(modname: str, attr: str) -> None:
    try:
        module = __import__(f"{__name__}.{modname}", fromlist=[attr])
        globals()[attr] = getattr(module, attr)
        __all__.append(attr)
    except Exception as e:  # noqa: BLE001 — intentional, see docstring
        print(
            f"[sc_reconstruction.decoders] {attr} unavailable: {type(e).__name__}: {e}",
            file=_sys.stderr,
        )


_try_import("reconmlp", "ReconMLPDecoder")
_try_import("recontransformer", "ReconTransformerDecoder")
