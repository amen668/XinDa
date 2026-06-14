"""Multi-dialect XML frontends for the translation pipeline.

`FormatProfile` lifts the LaTeXML-specific tag/namespace constants out of the
extract/apply logic so a second XML dialect (JATS, the journal-publishing
standard) reuses the same placeholder contract and translation engine.
"""

from xinda.formats.profiles import (
    JATS_PROFILE,
    LTX_PROFILE,
    FormatProfile,
    get_profile,
)
from xinda.formats.xml_units import apply_units, extract_units

__all__ = [
    "FormatProfile",
    "JATS_PROFILE",
    "LTX_PROFILE",
    "apply_units",
    "extract_units",
    "get_profile",
]
