"""Format-profile parity: the generalized `formats.xml_units` extractor, driven by
`LTX_PROFILE`, must reproduce the validated LaTeXML `extract._parse_xml` byte-for-byte.

This is what licenses the claim that the JATS frontend reuses *the same* extraction
contract rather than a divergent fork: if the profile-driven code matches the
load-bearing stage on real LaTeXML output, then swapping in `JATS_PROFILE` only
changes the tag/namespace constants, not the logic.

Needs a LaTeXML XML under workspace/ (same precondition as test_extract_parity.py);
skipped if none is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from xinda.formats import LTX_PROFILE, extract_units
from xinda.pipeline.stages.extract import _parse_xml


def _find_latexml_xml() -> Path | None:
    for cand in Path("workspace").glob("*/*/*.xml"):
        try:
            root = etree.parse(str(cand)).getroot()
        except etree.XMLSyntaxError:
            continue
        if isinstance(root.tag, str) and "LaTeXML" in root.tag:
            return cand
    return None


_XML = _find_latexml_xml()


def _sig(units: list[dict]) -> list[tuple]:
    """Identity tuple per unit: the load-bearing extraction outputs."""
    return [
        (
            u["xpath"],
            u["src_text"],
            u["src_plain"],
            tuple(sorted(u["placeholders"].values())),
            tuple(sorted(u["special_chars"].items())),
        )
        for u in units
    ]


@pytest.mark.skipif(_XML is None, reason="no LaTeXML XML in workspace/")
def test_generalized_extractor_matches_legacy_stage():
    assert _XML is not None
    _legacy_sections, legacy_units = _parse_xml(_XML)

    tree = etree.parse(str(_XML))
    _gen_sections, gen_units = extract_units(tree, LTX_PROFILE)

    assert len(gen_units) == len(legacy_units), (
        f"unit count differs: generalized={len(gen_units)} legacy={len(legacy_units)}"
    )
    assert _sig(gen_units) == _sig(legacy_units), (
        "generalized extractor diverged from the validated LaTeXML stage"
    )


@pytest.mark.skipif(_XML is None, reason="no LaTeXML XML in workspace/")
def test_generalized_sections_match_legacy_stage():
    assert _XML is not None
    legacy_sections, _ = _parse_xml(_XML)
    tree = etree.parse(str(_XML))
    gen_sections, _ = extract_units(tree, LTX_PROFILE)

    def ssig(secs: list[dict]) -> list[tuple]:
        return [(s["xpath"], s["depth"], s["heading"]) for s in secs]

    assert ssig(gen_sections) == ssig(legacy_sections)
