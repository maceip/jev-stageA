"""Dataset-audit helper contracts (counts only; never exfiltrate matches)."""

from audit_redo_dataset import luhn_ok, scan_text, script_bucket


def test_luhn_distinguishes_card_like_from_random_digits():
    assert luhn_ok("4111111111111111") is True
    assert luhn_ok("4111111111111112") is False
    assert luhn_ok("12345") is False


def test_scan_text_counts_patterns_without_returning_text():
    hits = scan_text("contact a@b.co card 4111 1111 1111 1111 key sk-abcdefghijklmnop")
    assert hits["email"] == 1
    assert hits["card_luhn"] == 1
    assert hits["api_key_like"] == 1
    assert scan_text("ordinary meeting notes") == {}


def test_script_bucket_classifies_dominant_script():
    assert script_bucket("hello world") == "latin"
    assert script_bucket("привет мир") == "cyrillic"
    assert script_bucket("你好世界") == "cjk"
    assert script_bucket("12345") == "no_alpha"
