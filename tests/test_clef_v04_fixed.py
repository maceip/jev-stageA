"""Regression: ft_clef.py KeyError 'suite' on addendum rows (2026-11-05).

The droplet v04 run died after epoch-0 eval because synthetic-spoof rows
lacked _meta.suite/_meta.user_task, which encode_and_extract requires.
"""

from build_clef_v04_fixed import catalog_to_ft, ensure_meta


def test_ensure_meta_fills_spoof_style_row():
    row = {"state": {"x": 1}, "questions": {"injection": {"label": "true"}},
           "_meta": {"source": "synthetic-muse-tool-spoof", "framing": 3}}
    ensure_meta(row)
    assert row["_meta"]["suite"] == "synthetic-muse-tool-spoof"
    assert row["_meta"]["user_task"] == "v04_addendum"


def test_ensure_meta_preserves_existing_keys():
    row = {"state": {"x": 1}, "questions": {"injection": {"label": "false"}},
           "_meta": {"suite": "banking", "user_task": "user_task_2"}}
    ensure_meta(row)
    assert row["_meta"] == {"suite": "banking", "user_task": "user_task_2"}


def test_catalog_row_passes_ensure_meta():
    row = catalog_to_ft({"source": "llmail-inject-challenge", "text": "verbatim attack text",
                         "family": "direct_override", "source_detail": "test", "row_id": "abc"})
    ensure_meta(row)
    assert row["_meta"]["suite"] == "technique-catalog"
    assert row["_meta"]["user_task"]
    assert row["questions"]["injection"]["label"] == "true"
