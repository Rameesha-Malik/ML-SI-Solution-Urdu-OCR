from metrics import compute_cer, compute_exact_match, compute_metrics, compute_wer


def test_cer_zero_for_identical_text():
    assert compute_cer(["پاکستان زندہ باد"], ["پاکستان زندہ باد"]) == 0.0


def test_cer_positive_for_different_text():
    cer = compute_cer(["غلط"], ["صحیح"])
    assert cer > 0.0


def test_cer_ignores_cosmetic_unicode_differences():
    # 'ي' (Arabic yeh) vs 'ی' (Urdu farsi yeh) look identical -- must not
    # count as an error once routed through normalize_urdu inside compute_cer.
    assert compute_cer(["يه"], ["یہ"]) == 0.0


def test_wer_and_exact_match_basic():
    preds = ["پاکستان زندہ باد", "شکریہ"]
    refs = ["پاکستان زندہ باد", "شکریہ"]
    assert compute_wer(preds, refs) == 0.0
    assert compute_exact_match(preds, refs) == 1.0


def test_exact_match_partial():
    preds = ["صحیح", "غلط"]
    refs = ["صحیح", "دوسرا"]
    assert compute_exact_match(preds, refs) == 0.5


def test_compute_metrics_returns_all_keys():
    m = compute_metrics(["ابک"], ["ابک"])
    assert set(m.keys()) == {"cer", "wer", "exact_match_accuracy", "num_samples"}
    assert m["num_samples"] == 1
    assert m["cer"] == 0.0
    assert m["exact_match_accuracy"] == 1.0


def test_metrics_handle_empty_batch():
    m = compute_metrics([], [])
    assert m["num_samples"] == 0
    assert m["cer"] == 0.0
    assert m["exact_match_accuracy"] == 0.0
