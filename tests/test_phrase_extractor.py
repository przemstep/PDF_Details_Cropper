from app.phrase_extractor import extract_core_phrases


def _phrases(text: str) -> set[str]:
    return {c.normalized_phrase for c in extract_core_phrases(text)}


def test_noise_filtered() -> None:
    p = _phrases("REI120\n+9.56\n150\nD=40 mm")
    assert not p


def test_strip_tail_pl() -> None:
    p = _phrases("STROP ŻELBETOWY WG PROJEKTU KONSTRUKCJI")
    assert "strop zelbetowy" in p


def test_strip_tail_en() -> None:
    p = _phrases("RC SLAB REF. TO STRUCT. ENG. DESIGN")
    assert "rc slab" in p


def test_mineral_wool() -> None:
    p = _phrases("WEŁNA MINERALNA WG SPECYFIKACJI\nMINERAL WOOL ACC. TO SPECIFICATION")
    assert any(x in p for x in {"welna mineralna", "wena mineralna"})
    assert "mineral wool" in p


def test_dimension_removed() -> None:
    p = _phrases("SŁUPEK BALUSTRADY ZE STALI NIERDZEWNEJ D=40 mm")
    assert any(x in p for x in {"slupek balustrady", "supek balustrady"})
    assert not any("d 40 mm" in x for x in p)


def test_english_rail() -> None:
    p = _phrases("STAINLESS STEEL TOP RAIL")
    assert "stainless steel top rail" in p
    assert "top rail" in p
