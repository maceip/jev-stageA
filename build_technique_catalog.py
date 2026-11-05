"""Build a verbatim prompt-injection technique catalog (>=100 unique attacks).

Pick-up of the WorkBuddy task halted by credits (2026-11-04): mine the
papers and public corpora for REAL prompt-injection examples before any
training. Sources are local, previously license-ledgered corpora plus
deepset/prompt-injections. HackAPrompt is HF-gated (403) and recorded as
blocked. Attack texts are extracted VERBATIM; technique families follow
the CMC 2025.074081 survey axes (direct/indirect vectors; leakage /
hijacking / privilege / exfiltration objectives) refined by LLMail-Inject
(arXiv:2506.09956) observed patterns. The 4 user-supplied PDFs live only
in the WorkBuddy cloud session (paper1_emnlp = LLMail-Inject,
paper2_cmc = CMC survey, paper3/4 arXiv); both identified papers were
re-fetched from arXiv/publisher for this build.

Outputs (addenda only, no training, frozen sets untouched):
  technique_catalog_100.jsonl        all tiers, per-row provenance
  technique_catalog_trainable.jsonl  train-ok license tiers only
  technique_catalog_canonical.jsonl  canonical redo rows (trainable)
TensorTrust (no license) and BIPIA/InjecAgent (prior eval-only tier)
are catalogued but excluded from the trainable file.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from redo_v2 import canonical_row, normalized_text_key

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "data/external"
OUT = EXT / "redo_v2_out"
SEED = 20261104

# Ordered family rules: first strong marker wins (checked top-down).
FAMILY_RULES: list[tuple[str, re.Pattern]] = [
    ("cot_hijack", re.compile(r"parse input:|identify tool:|extract arguments:|execute only step|simulated (cot|chain)|step 4", re.I)),
    ("code_injection_payload", re.compile(r"import (http|requests|os|sys|socket|subprocess)|```\s*(python|bash|sh|javascript)|code (section|block) (in|to) (your|the) (implementation|response)", re.I)),
    ("unicode_obfuscation", re.compile(r"[\u200b\u200c\uff01-\uff5e]|\\u200b|e with 3|replace the letter", re.I)),
    ("encoding_split_reassembly", re.compile(r"base64|decode|combine|1\+2|initial is|spell (out|ing)|first letter of each|acrostic", re.I)),
    ("tool_call_forgery", re.compile(r'"tool_call"\s*:|send_email|tool[_\s]?call|function[_\s]?call|arguments"\s*:\s*\{', re.I)),
    ("fake_system_admin", re.compile(r"<\|?system|begin system|admin logged|system auto generated|higher authority|developer mode|system override", re.I)),
    ("fake_user_turn", re.compile(r"<\|user\|>|<user>|begin user|\[user\]|human:", re.I)),
    ("fake_assistant_prefill", re.compile(r"<\|assistant\|>|<assistant>|assistant:\s*(sure|okay|certainly|the email|i (have|will))", re.I)),
    ("delimiter_escape", re.compile(r"<\|end\|>|```|\"\"\"\"|end of (the )?(email|document|prompt)|\{the body (starts|ends)|prompt (start|end)", re.I)),
    ("authority_social_engineering", re.compile(r"\bboss\b|\bmanager\b|microsoft ai|compliance|audit trail|capability enhancement|upgraded your system|urgent|immediately|as soon as possible", re.I)),
    ("direct_override", re.compile(r"ignore (all |the |any )?(previous|prior|above|earlier)|disregard|forget (all|your|the) (previous|instructions)|do not follow", re.I)),
    ("roleplay_persona", re.compile(r"act as|pretend (to be|you)|you are now|roleplay|persona|character of|\bDAN\b", re.I)),
    ("secret_extraction", re.compile(r"password|secret|system prompt|your (initial )?instructions|reveal|repeat (the|your)|output (the|your) (code|word|key)", re.I)),
    ("retrieval_relevance_stuffing", re.compile(r"(.{12,60})\1{2,}", re.S)),
]


def classify(text: str, default: str) -> str:
    for family, rx in FAMILY_RULES:
        if rx.search(text):
            return family
    return default


def clean(text: str, cap: int = 4000) -> str:
    return " ".join(str(text).split())[:cap].strip()


class Catalog:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.seen: set[str] = set()

    def add(self, text: str, *, source: str, tier: str, license_name: str, kind_hint: str,
            default_family: str, source_detail: str) -> bool:
        text = clean(text)
        if len(text) < 12:
            return False
        key = normalized_text_key(text)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.rows.append({
            "text": text, "family": classify(text, default_family), "source": source,
            "tier": tier, "license": license_name, "kind_hint": kind_hint,
            "source_detail": source_detail,
            "row_id": hashlib.sha256(key.encode()).hexdigest(),
        })
        return True


def load_llmail(cat: Catalog, quota: int = 46) -> Counter:
    got: Counter = Counter()
    base = ROOT / "data/redo/raw/microsoft__llmail-inject-challenge/data"
    rng = random.Random(SEED)
    per_family_cap = 8
    fam_count: Counter = Counter()
    for phase in ("labelled_unique_submissions_phase1.json", "labelled_unique_submissions_phase2.json"):
        data = json.loads((base / phase).read_text())
        items = [(k, v) for k, v in data.items()
                 if str((v or {}).get("attack_attempt", "")).lower() == "true"]
        rng.shuffle(items)
        rare = {"cot_hijack", "encoding_split_reassembly"}
        items.sort(key=lambda kv: 0 if classify(clean(kv[0]), "x") in rare else 1)
        for text, _v in items:
            if sum(got.values()) >= quota:
                break
            fam = classify(clean(text), "indirect_email_injection")
            if fam_count[fam] >= per_family_cap:
                continue
            if cat.add(text, source="llmail-inject-challenge", tier="train_ok_mit", license_name="MIT",
                       kind_hint="tool_output_filter", default_family="indirect_email_injection",
                       source_detail=f"microsoft/llmail-inject-challenge {phase} (arXiv:2506.09956)"):
                fam_count[fam] += 1
                got[fam] += 1
    return got


def load_gandalf(cat: Catalog, quota: int = 14) -> Counter:
    import pyarrow.parquet as pq
    got: Counter = Counter()
    fam_count: Counter = Counter()
    path = EXT / "gandalf_ignore_instructions/data/train-00000-of-00001-ded53be747ff55cd.parquet"
    texts = pq.read_table(path).column("text").to_pylist()
    rng = random.Random(SEED + 1)
    rows = list(texts)
    rng.shuffle(rows)
    for text in rows:
        if sum(got.values()) >= quota:
            break
        fam = classify(clean(str(text)), "secret_extraction")
        if fam_count[fam] >= 6:
            continue
        if cat.add(str(text), source="gandalf_ignore_instructions", tier="train_ok_mit", license_name="MIT",
                   kind_hint="direct_prompt_injection", default_family="secret_extraction",
                   source_detail="Lakera/gandalf_ignore_instructions train split"):
            fam_count[fam] += 1
            got[fam] += 1
    return got


def load_deepset(cat: Catalog, quota: int = 14) -> Counter:
    import pyarrow.parquet as pq
    got: Counter = Counter()
    fam_count: Counter = Counter()
    rng = random.Random(SEED + 2)
    for split in ("train-00000-of-00001-9564e8b05b4757ab", "test-00000-of-00001-701d16158af87368"):
        path = EXT / f"pi-extra/deepset/data/{split}.parquet"
        if not path.is_file():
            continue
        t = pq.read_table(path).to_pylist()
        rows = [r for r in t if int(r["label"]) == 1]
        rng.shuffle(rows)
        for r in rows:
            if sum(got.values()) >= quota:
                break
            fam = classify(clean(r["text"]), "direct_override")
            if fam_count[fam] >= 6:
                continue
            if cat.add(r["text"], source="deepset-prompt-injections", tier="train_ok_ccby",
                       license_name="CC BY 4.0 (per HF dataset card)", kind_hint="direct_prompt_injection",
                       default_family="direct_override", source_detail="deepset/prompt-injections (Perez & Ribeiro lineage)"):
                fam_count[fam] += 1
                got[fam] += 1
    return got


def load_satml(cat: Catalog, quota: int = 14) -> Counter:
    got: Counter = Counter()
    fam_count: Counter = Counter()
    rng = random.Random(SEED + 3)
    for name in ("chat.was_successful_secret_extraction.50.json", "chat.50.json"):
        data = json.loads((EXT / f"ctf-satml24/{name}").read_text())
        rng.shuffle(data)
        for chat in data:
            if sum(got.values()) >= quota:
                break
            for msg in chat.get("history") or []:
                if sum(got.values()) >= quota:
                    break
                if msg.get("role") != "user":
                    continue
                text = str(msg.get("content") or "")
                fam = classify(clean(text), "secret_extraction")
                if fam_count[fam] >= 6:
                    continue
                if cat.add(text, source="ctf-satml24", tier="train_ok_mit", license_name="MIT",
                           kind_hint="direct_prompt_injection", default_family="secret_extraction",
                           source_detail=f"ethz-spylab/ctf-satml24 {name}"):
                    fam_count[fam] += 1
                    got[fam] += 1
    return got


def load_bipia(cat: Catalog, quota: int = 16) -> Counter:
    got: Counter = Counter()
    rng = random.Random(SEED + 4)
    for name, fam_default, kind in (("text_attack_train.json", "task_redirect_qa", "tool_output_filter"),
                                    ("code_attack_train.json", "code_injection_payload", "tool_output_filter")):
        data = json.loads((EXT / f"repos/BIPIA/benchmark/{name}").read_text())
        cats = sorted(data)
        rng.shuffle(cats)
        file_got = 0
        for category in cats:
            if file_got >= quota // 2:
                break
            items = list(data[category])
            rng.shuffle(items)
            for text in items[:1]:
                if cat.add(str(text), source="bipia", tier="eval_only_prior_protocol", license_name="MIT",
                           kind_hint=kind, default_family=fam_default,
                           source_detail=f"microsoft/BIPIA benchmark/{name} category={category}"):
                    got[classify(clean(str(text)), fam_default)] += 1
                    file_got += 1
    return got


def load_injecagent(cat: Catalog, quota: int = 12) -> Counter:
    got: Counter = Counter()
    rng = random.Random(SEED + 5)
    fam_count: Counter = Counter()
    for name in ("attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"):
        lines = (EXT / f"repos/InjecAgent/data/{name}").read_text().splitlines()
        rows = [json.loads(x) for x in lines if x.strip()]
        rng.shuffle(rows)
        for r in rows:
            if sum(got.values()) >= quota:
                break
            atype = str(r.get("Attack Type") or "indirect_tool_harm").strip().lower().replace(" ", "_")
            fam = f"indirect_tool_harm:{atype}"
            if fam_count[fam] >= 3:
                continue
            if cat.add(str(r.get("Attacker Instruction") or ""), source="injecagent",
                       tier="eval_only_prior_protocol", license_name="see repo LICENSE",
                       kind_hint="tool_output_filter", default_family=fam,
                       source_detail=f"uiuc-kang-lab/InjecAgent data/{name} type={r.get('Attack Type')}"):
                fam_count[fam] += 1
                got[fam] += 1
    return got


def load_tensortrust(cat: Catalog, quota: int = 12) -> Counter:
    got: Counter = Counter()
    fam_count: Counter = Counter()
    rng = random.Random(SEED + 6)
    path = EXT / "repos/tensor-trust-data/raw-data/v2/raw_dump_attacks.jsonl.bz2"
    rows = []
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 60000:
                break
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rng.shuffle(rows)
    text_keys = ("attack", "attack_prompt", "prompt", "text", "input")
    for r in rows:
        if sum(got.values()) >= quota:
            break
        text = next((str(r[k]) for k in text_keys if r.get(k)), "")
        if not text and r.get("attack") is None:
            # fall back: longest string value in the record
            strs = [str(v) for v in r.values() if isinstance(v, str)]
            text = max(strs, key=len) if strs else ""
        fam = classify(clean(text), "secret_extraction")
        if fam_count[fam] >= 5:
            continue
        if cat.add(text, source="tensor-trust", tier="excluded_no_license", license_name="NONE FOUND (excluded)",
                   kind_hint="direct_prompt_injection", default_family="secret_extraction",
                   source_detail="tensor-trust-data raw-data/v2 (ICLR 2024, Toyer et al.)"):
            fam_count[fam] += 1
            got[fam] += 1
    return got


def main() -> None:
    cat = Catalog()
    per_source = {}
    for name, fn in [("llmail", load_llmail), ("gandalf", load_gandalf), ("deepset", load_deepset),
                     ("satml24", load_satml), ("bipia", load_bipia), ("injecagent", load_injecagent),
                     ("tensortrust", load_tensortrust)]:
        try:
            per_source[name] = dict(fn(cat))
        except Exception as exc:
            per_source[name] = {"ERROR": f"{type(exc).__name__}: {str(exc)[:160]}"}
    trainable = [r for r in cat.rows if r["tier"].startswith("train_ok")]
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fname, rows in [("technique_catalog_100.jsonl", cat.rows),
                        ("technique_catalog_trainable.jsonl", trainable)]:
        p = OUT / fname
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[fname] = {"rows": len(rows), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    can_path = OUT / "technique_catalog_canonical.jsonl"
    with can_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(trainable):
            f.write(json.dumps(canonical_row(
                source=r["source"], source_id=f"technique-catalog-{i}", tier="B_real_human",
                kind=r["kind_hint"], text=r["text"], label=True,
                group_keys={"participant": r["source"], "template_cluster": normalized_text_key(r["family"])[:16]},
                supported_tasks=[r["kind_hint"]],
                metadata={"family": r["family"], "license": r["license"], "source_detail": r["source_detail"],
                          "label_source": "verbatim_public_corpus"}), ensure_ascii=False) + "\n")
    paths["technique_catalog_canonical.jsonl"] = {
        "rows": len(trainable), "sha256": hashlib.sha256(can_path.read_bytes()).hexdigest()}
    families = Counter(r["family"] for r in cat.rows)
    summary = {"total_unique": len(cat.rows), "trainable": len(trainable),
               "families": dict(sorted(families.items())), "n_families": len(families),
               "per_source": per_source, "files": paths,
               "blocked_sources": ["hackaprompt/hackaprompt-dataset: HF gated 403 (access request required)"]}
    (OUT / "technique_catalog_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert len(cat.rows) >= 100, "catalog below 100 unique attacks"
    assert len(families) >= 12, "fewer than 12 technique families"


if __name__ == "__main__":
    main()
