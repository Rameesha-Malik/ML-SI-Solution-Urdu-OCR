from text_utils import is_urdu_text, normalize_urdu


def test_normalize_letter_forms():
    # Arabic 'yeh' (U+064A) and 'heh' (U+0647) -> Urdu canonical forms.
    assert normalize_urdu("يه") == "یہ"


def test_normalize_digits_to_ascii():
    assert normalize_urdu("قیمت ٥٠٠ روپے", digits="ascii") == "قیمت 500 روپے"
    assert normalize_urdu("قیمت ۵۰۰ روپے", digits="ascii") == "قیمت 500 روپے"


def test_normalize_digits_to_urdu():
    out = normalize_urdu("قیمت 500 روپے", digits="urdu")
    assert out == "قیمت ۵۰۰ روپے"


def test_normalize_digits_keep():
    out = normalize_urdu("قیمت 500 روپے", digits="keep")
    assert "500" in out


def test_normalize_strips_diacritics_by_default():
    vocalised = "مَكَّةَ"  # heavily vocalised
    out = normalize_urdu(vocalised)
    # no combining diacritic marks should remain
    assert not any(0x064B <= ord(c) <= 0x0652 for c in out)


def test_normalize_keeps_diacritics_when_disabled():
    vocalised = "بَ"
    out = normalize_urdu(vocalised, strip_diacritics=False)
    assert any(0x064B <= ord(c) <= 0x0652 for c in out)


def test_normalize_collapses_whitespace():
    assert normalize_urdu("  آج   بروز  جمعہ  ") == "آج بروز جمعہ"


def test_normalize_strips_tatweel():
    assert "ـ" not in normalize_urdu("جمعـــہ")


def test_normalize_empty_and_none():
    assert normalize_urdu("") == ""
    assert normalize_urdu(None) == ""


def test_is_urdu_text_true_for_urdu():
    assert is_urdu_text("پاکستان زندہ باد") is True


def test_is_urdu_text_false_for_english():
    assert is_urdu_text("Hello world, this is English") is False


def test_is_urdu_text_false_for_empty():
    assert is_urdu_text("") is False
    assert is_urdu_text("   ") is False
