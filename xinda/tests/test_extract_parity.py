"""Pure-function parity test: new _parse_xml vs old extract_translation_items.

Doesn't touch the DB. Run with:
    pytest xinda/tests/test_extract_parity.py -v

Expects an existing XML at workspace/2503.15129/<ts>/<stem>.xml — point at
any LaTeXML output you have. The test ensures the new extractor produces
the same number of translation units as the old `extract_html` function.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The v1 extract_html lived in legacy/ (removed from the open-source tree); if a
# checkout still has it, put it on sys.path so the parity test can run.
_LEGACY = Path(__file__).resolve().parents[2] / "legacy"
if _LEGACY.is_dir():
    sys.path.insert(0, str(_LEGACY))

from xinda.pipeline.stages.extract import _parse_xml


def _find_test_xml() -> Path | None:
    candidates = list(Path("workspace").glob("*/*/*.xml"))
    return candidates[0] if candidates else None


@pytest.mark.skipif(_find_test_xml() is None, reason="no test XML in workspace/")
def test_unit_count_matches_legacy():
    extract_html = pytest.importorskip("extract_html", reason="legacy/ v1 code not present")
    extract_translation_items = extract_html.extract_translation_items

    xml = _find_test_xml()
    assert xml is not None
    legacy_items = extract_translation_items(str(xml))
    sections, units = _parse_xml(xml)

    # New extractor includes section_heading + caption + abstract too,
    # so ≥ legacy is the right invariant. Legacy gathered all <ltx:p|title|caption>
    # so the two should match closely.
    assert len(units) >= len(legacy_items), (
        f"new extractor lost units: new={len(units)} legacy={len(legacy_items)}"
    )
    # tolerate small differences but flag anomalies
    diff = len(units) - len(legacy_items)
    assert diff < 20, f"unit-count drift too large: +{diff}"
