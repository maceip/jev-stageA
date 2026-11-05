"""Canonical dataset contracts for the CLEF/SecJev v2 redo.

The module is deliberately source- and label-conservative:

* a row may support only tasks its source can actually label;
* direct-injection corpora never become action-authorization examples by
  renaming alone;
* splits are assigned from stable group hashes so every row sharing a
  participant/template group stays on one side of a split;
* external benchmarks used for reporting are marked test-only in the
  registry and must not be folded into training by a converter.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping

ALLOWED_KINDS = {"tool_output_filter", "action_authorization_guard", "direct_prompt_injection"}
SPLIT_SEED = "redo-v2-20261103"

SOURCE_REGISTRY = [
    {
        "name": "llmail-inject-challenge",
        "url": "https://huggingface.co/datasets/microsoft/llmail-inject-challenge",
        "revision": "1063bdf01ec8762b812d5e06ee768a06faa5a6f7",
        "license": "mit",
        "tier": "A_human_adaptive",
        "use": "train_dev_and_locked_test_after_group_split",
        "supported_tasks": ["tool_output_filter"],
        "split_groups": ["team_id", "scenario", "template_cluster"],
        "local_path": "data/external/llmail-inject-challenge",
        "status": "downloaded_pinned_revision",
    },
    {
        "name": "ctf-satml24",
        "url": "https://huggingface.co/datasets/ethz-spylab/ctf-satml24",
        "revision": "11426babe61ee16b0efe7b55e0430359e0fc4235",
        "license": "mit",
        "tier": "A_human_adaptive",
        "use": "direct_injection_train_dev_and_external_test",
        "supported_tasks": ["direct_prompt_injection"],
        "split_groups": ["user", "defense", "template_cluster"],
        "local_path": "data/external/ctf-satml24",
        "status": "downloaded_pinned_revision",
    },
    {
        "name": "gandalf_ignore_instructions",
        "url": "https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
        "revision": "04737b65e90a6794ec227012e4a255a7def6344b",
        "license": "mit",
        "tier": "A_human_adaptive",
        "use": "seed_and_direct_injection_eval",
        "supported_tasks": ["direct_prompt_injection"],
        "split_groups": ["participant", "template_cluster"],
        "local_path": "data/external/gandalf_ignore_instructions",
        "status": "downloaded_pinned_revision",
    },
    {
        "name": "tensor-trust",
        "url": "https://tensortrust.ai/paper",
        "revision": "747a75e096761ebc01bd3970158827326b4add23",
        "license": "no_license_file_in_data_repo",
        "tier": "A_human_adaptive",
        "use": "quarantined_until_primary_files_and_terms_are_verified",
        "supported_tasks": ["direct_prompt_injection"],
        "split_groups": ["defense", "attack_strategy", "template_cluster"],
        "local_path": "data/external/repos/tensor-trust-data",
        "status": "cloned_quarantined_no_license",
    },
    {
        "name": "bipia",
        "url": "https://github.com/microsoft/BIPIA",
        "revision": "a004b69ec0dd446e0afd461d98cb5e96e120a5d0",
        "license": "mit_code_component_data_terms_separate",
        "tier": "B_benchmark_context",
        "use": "external_context_eval_or_licensed_components_only",
        "supported_tasks": ["tool_output_filter"],
        "split_groups": ["task", "context_source", "template_cluster"],
        "local_path": "data/external/repos/BIPIA",
        "status": "cloned_shallow_component_audit_open",
    },
    {
        "name": "injecagent",
        "url": "https://github.com/uiuc-kang-lab/InjecAgent",
        "revision": "f19c9f2c79a41046eb13c03c51a24c567a8ffa07",
        "license": "mit",
        "tier": "B_benchmark_synthetic",
        "use": "external_test_only",
        "supported_tasks": ["tool_output_filter", "action_authorization_guard"],
        "split_groups": ["user_tool", "attacker_tool"],
        "local_path": "data/external/repos/InjecAgent",
        "status": "cloned_shallow_external_test_only",
    },
    {
        "name": "pint",
        "url": "https://github.com/lakeraai/pint-benchmark",
        "revision": None,
        "license": "benchmark_code_mit_dataset_terms_separate",
        "tier": "B_benchmark_mixed",
        "use": "external_test_only_if_access_terms_allow",
        "supported_tasks": ["tool_output_filter", "direct_prompt_injection"],
        "split_groups": ["category", "language"],
        "local_path": None,
        "status": "not_acquired",
    },
]


def normalized_text_key(text: str) -> str:
    """Stable exact/near-duplicate key for conservative leakage checks."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"[^\w ]+", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_group_split(group: str, seed: str = SPLIT_SEED) -> str:
    """Assign a whole group to train/dev/test from a stable 70/15/15 hash."""
    bucket = int(hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest()[:8], 16) % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "dev"
    return "test"


def canonical_row(
    *,
    source: str,
    source_id: str,
    tier: str,
    kind: str,
    text: str,
    label: bool,
    group_keys: Mapping[str, str],
    supported_tasks: Iterable[str],
    metadata: Mapping | None = None,
) -> dict:
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unsupported kind: {kind}")
    if not text or not text.strip():
        raise ValueError("canonical row text must be non-empty")
    groups = dict(group_keys)
    for required in ("participant", "template_cluster"):
        if not str(groups.get(required, "")).strip():
            raise ValueError(f"group_keys.{required} is required")
    tasks = list(dict.fromkeys(supported_tasks))
    if not tasks or not set(tasks).issubset(ALLOWED_KINDS):
        raise ValueError(f"unsupported supported_tasks: {tasks}")
    identity = {
        "source": source,
        "source_id": str(source_id),
        "kind": kind,
        "text_key": normalized_text_key(text),
    }
    return {
        "row_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "source": source,
        "source_id": str(source_id),
        "tier": tier,
        "kind": kind,
        "text": text,
        "label": bool(label),
        "group_keys": groups,
        "supported_tasks": tasks,
        "metadata": dict(metadata or {}),
    }


def _parse_objectives(value) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, Mapping) else {"raw": parsed}
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def convert_llmail_raw(row: Mapping) -> dict:
    """Convert one LLMail raw submission to a Spot-1/filter row.

    LLMail objectives describe the submitted attack outcome chain. The raw
    row alone does not identify a concrete proposed tool call, so this
    converter deliberately does not claim action-authorization support.
    """
    subject = str(row.get("subject") or "")
    body = str(row.get("body") or "")
    text = f"Subject: {subject}\n\n{body}".strip()
    team = str(row.get("team_id") or "unknown-team")
    return canonical_row(
        source="llmail-inject-challenge",
        source_id=str(row.get("RowKey") or row.get("job_id") or ""),
        tier="A_human_adaptive",
        kind="tool_output_filter",
        text=text,
        label=True,
        group_keys={
            "participant": team,
            "template_cluster": normalized_text_key(text)[:16],
            "scenario": str(row.get("scenario") or "unknown"),
        },
        supported_tasks=["tool_output_filter"],
        metadata={
            "objectives": _parse_objectives(row.get("objectives")),
            "scenario": row.get("scenario"),
            "scheduled_time": row.get("scheduled_time"),
            "label_rationale": "human-submitted LLMail attack email",
        },
    )


def _history_text(history) -> str:
    """Attacker-side text only; assistant turns can leak the outcome label."""
    parts = []
    if isinstance(history, str):
        return history
    if isinstance(history, Iterable):
        for turn in history:
            if isinstance(turn, str):
                parts.append(turn)
            elif isinstance(turn, Mapping):
                role = str(turn.get("role") or "").lower()
                if role and role not in {"user", "attacker", "human"}:
                    continue
                parts.append(str(turn.get("content") or turn.get("text") or turn.get("message") or ""))
    return "\n".join(p for p in parts if p)


def _ref_id(value) -> str:
    if isinstance(value, Mapping):
        team = value.get("team")
        if team:
            return str(team)
        oid = value.get("$id")
        if isinstance(oid, Mapping):
            return str(oid.get("$oid") or "unknown")
        return str(oid or "unknown")
    return str(value or "unknown")


def convert_satml_chat(row: Mapping) -> dict:
    text = _history_text(row.get("history"))
    participant = _ref_id(row.get("user")) if row.get("user") else str(row.get("_id") or "unknown-user")
    defense = _ref_id(row.get("defense")) if row.get("defense") else "unknown-defense"
    return canonical_row(
        source="ctf-satml24",
        source_id=str(row.get("_id") or ""),
        tier="A_human_adaptive",
        kind="direct_prompt_injection",
        text=text,
        label=bool(row.get("is_attack")),
        group_keys={
            "participant": participant,
            "template_cluster": normalized_text_key(text)[:16],
            "defense": defense,
        },
        supported_tasks=["direct_prompt_injection"],
        metadata={
            "model": row.get("model"),
            "was_successful_secret_extraction": row.get("was_successful_secret_extraction"),
            "label_rationale": "SaTML is_attack field; direct secret-extraction setting only",
        },
    )


def convert_gandalf(row: Mapping, source_id: str) -> dict:
    text = str(row.get("text") or "")
    return canonical_row(
        source="gandalf_ignore_instructions",
        source_id=source_id,
        tier="A_human_adaptive",
        kind="direct_prompt_injection",
        text=text,
        label=True,
        group_keys={
            "participant": "gandalf-public-sample",
            "template_cluster": normalized_text_key(text)[:16],
        },
        supported_tasks=["direct_prompt_injection"],
        metadata={"similarity": row.get("similarity")},
    )


def normalize_attack_attempt(value) -> bool | None:
    """Normalize LLMail labelled-file values; Unclear/junk returns None."""
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str):
        v = value.strip().strip('",').lower()
        if v == "true":
            return True
        if v == "false":
            return False
    return None


def convert_llmail_labelled(text: str, value: Mapping, phase: int) -> dict | None:
    """Convert one labelled-unique LLMail entry; None means dropped as Unclear."""
    label = normalize_attack_attempt(value.get("attack_attempt"))
    if label is None:
        return None
    cluster = normalized_text_key(text)[:16]
    return canonical_row(
        source="llmail-inject-challenge",
        source_id=f"labelled-phase{phase}-{cluster}",
        tier="A_human_adaptive",
        kind="tool_output_filter",
        text=text,
        label=label,
        group_keys={
            # Labelled files carry no team id; never invent one. The
            # per-text pseudo-participant records that limitation.
            "participant": f"llmail-labelled-unknown:{cluster}",
            "template_cluster": cluster,
            "scenario": f"phase{phase}",
        },
        supported_tasks=["tool_output_filter"],
        metadata={
            "label_source": "labelled_unique_submissions",
            "label_reason": value.get("reason"),
            "participant_known": False,
            "phase": phase,
        },
    )


def convert_benign_email(text: str, source_id: str) -> dict:
    cluster = normalized_text_key(text)[:16]
    return canonical_row(
        source="llmail-inject-challenge",
        source_id=source_id,
        tier="A_human_adaptive",
        kind="tool_output_filter",
        text=text,
        label=False,
        group_keys={"participant": "llmail-fp-tests", "template_cluster": cluster},
        supported_tasks=["tool_output_filter"],
        metadata={"label_source": "emails_for_fp_tests"},
    )


def _components(materialized: list[Mapping], row_level_kinds: frozenset = frozenset()) -> list[list[int]]:
    parent = list(range(len(materialized)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_seen: dict[tuple[str, str, str], int] = {}
    for idx, key in enumerate(materialized):
        groups = key.get("group_keys") or {}
        tokens = [("text", "normalized", str(key["text_key"]))]
        if str(key.get("kind")) in row_level_kinds:
            # Chat-level split unit: only identical texts stay together.
            # Participant overlap across splits must be disclosed by callers.
            for token in tokens:
                if token in first_seen:
                    union(idx, first_seen[token])
                else:
                    first_seen[token] = idx
            continue
        for field in ("participant", "template_cluster"):
            if not str(groups.get(field, "")).strip():
                raise ValueError(f"key lacks group_keys.{field}")
            tokens.append((str(key["source"]), field, str(groups[field])))
        for token in tokens:
            if token in first_seen:
                union(idx, first_seen[token])
            else:
                first_seen[token] = idx
    components: dict[int, list[int]] = {}
    for idx in range(len(materialized)):
        components.setdefault(find(idx), []).append(idx)
    return list(components.values())


def assign_splits_from_keys(keys: Iterable[Mapping], seed: str = SPLIT_SEED) -> dict[str, str]:
    """Map row_id -> split from lightweight key records (no row text needed)."""
    materialized = list(keys)
    homes: dict[str, str] = {}
    for indexes in _components(materialized):
        representative = min(str(materialized[i]["row_id"]) for i in indexes)
        split = stable_group_split(representative, seed=seed)
        for i in indexes:
            homes[str(materialized[i]["row_id"])] = split
    return homes


def assign_splits_stratified(keys: Iterable[Mapping], seed: str = SPLIT_SEED) -> dict[str, str]:
    """Component-disjoint splits balanced per (kind, label) signature.

    Hash-only assignment can strand every rare-label component in one split
    (observed: all 872 SaTML non-attacks in train). This greedy variant
    assigns largest components first to the split furthest below its
    per-signature target share. Deterministic for a fixed input order.
    """
    materialized = list(keys)
    components = _components(materialized, row_level_kinds=frozenset({"direct_prompt_injection"}))
    signatures = []
    totals: dict[tuple, int] = {}
    for indexes in components:
        sig: dict[tuple, int] = {}
        for i in indexes:
            k = (str(materialized[i].get("kind")), bool(materialized[i].get("label")))
            sig[k] = sig.get(k, 0) + 1
            totals[k] = totals.get(k, 0) + 1
        signatures.append(sig)
    ratios = {"train": 0.70, "dev": 0.15, "test": 0.15}
    targets = {s: {k: max(1.0, totals[k] * r) for k in totals} for s, r in ratios.items()}
    assigned: dict[str, dict[tuple, int]] = {s: {} for s in ratios}
    order = sorted(
        range(len(components)),
        key=lambda c: (-len(components[c]), min(str(materialized[i]["row_id"]) for i in components[c])),
    )
    homes: dict[str, str] = {}
    for c in order:
        def penalty(split: str) -> float:
            return sum(
                ((assigned[split].get(k, 0) + n) / targets[split][k]) ** 2
                for k, n in signatures[c].items()
            )

        split = min(("train", "dev", "test"), key=lambda s: (penalty(s), s))
        for k, n in signatures[c].items():
            assigned[split][k] = assigned[split].get(k, 0) + n
        for i in components[c]:
            homes[str(materialized[i]["row_id"])] = split
    return homes


def flatten_strings(value):
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield str(value)
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from flatten_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from flatten_strings(v)


def authorization_label(function: str, argv_strings, user_fns, inj_fns, inj_vals) -> int | None:
    """Corrected AgentDojo ground-truth label rule.

    2 (unauthorized) requires the call function in the injection ground
    truth AND an attacker-specific argument value in the call arguments.
    0 (authorized) requires the function in the user ground truth and not
    unauthorized. Everything else is ambiguous (None -> skip). The legacy
    builder labelled injection-only functions unauthorized even without an
    attacker value; that branch is the bug this rule removes.
    """
    argv = " ".join(str(v) for v in argv_strings)
    bad = function in inj_fns and any(str(v) and str(v) in argv for v in inj_vals)
    if bad:
        return 2
    if function in user_fns:
        return 0
    return None


def legacy_authorization_label(function: str, argv_strings, user_fns, inj_fns, inj_vals) -> int | None:
    """The pre-redo rule, kept only to quantify relabelling impact."""
    argv = " ".join(str(v) for v in argv_strings)
    bad = function in inj_fns and (function not in user_fns or any(str(v) and str(v) in argv for v in inj_vals))
    if bad:
        return 2
    if function in user_fns:
        return 0
    return None


def convert_action_point(meta: Mapping, payload: Mapping, label: int, source_id: str) -> dict:
    """Canonical Spot-2 row from one score.decision_points action payload."""
    from score import secjev_action_request

    req = secjev_action_request(payload, with_content=True)
    text = json.dumps(req["state"], ensure_ascii=False, sort_keys=True)
    suite = str(meta.get("suite") or "unknown")
    user_task = str(meta.get("user_task") or "unknown")
    return canonical_row(
        source="agentdojo-recorded-runs",
        source_id=source_id,
        tier="C_local_recorded",
        kind="action_authorization_guard",
        text=text,
        label=(label != 0),
        group_keys={
            "participant": f"{suite}:{user_task}",
            "template_cluster": str(meta.get("injection_task") or "benign"),
            "suite": suite,
        },
        supported_tasks=["action_authorization_guard"],
        metadata={
            "label_raw": label,
            "file": meta.get("file"),
            "suite": suite,
            "user_task": user_task,
            "injection_task": meta.get("injection_task"),
            "function": payload["call"]["function"],
            "label_rule": "corrected_v2_attacker_value_required",
        },
    )


def convert_benign_tool_output(text: str, source_id: str, participant: str) -> dict:
    """Benign tool output observed in a no-attack episode (hard negative)."""
    cluster = normalized_text_key(text)[:16]
    return canonical_row(
        source="agentdojo-benign-outputs",
        source_id=source_id,
        tier="C_local_recorded",
        kind="tool_output_filter",
        text=text,
        label=False,
        group_keys={"participant": participant, "template_cluster": cluster},
        supported_tasks=["tool_output_filter"],
        metadata={"label_source": "benign_episode_seen_output"},
    )


def convert_derived_action_row(row: Mapping, corrected_label: int, legacy_label: int, source_id: str) -> dict:
    """Canonical Spot-2 row from a legacy derived ft/ft_c row, relabelled.

    The legacy label is preserved in metadata so relabelling impact stays
    auditable; the canonical label is always the corrected one.
    """
    state = row["state"]
    seen = state.get("untrusted_tool_outputs_seen") or []
    view = "content" if any(isinstance(s, Mapping) and "output" in s for s in seen) else "metadata"
    meta = row.get("_meta") or {}
    suite = str(meta.get("suite") or "unknown")
    user_task = str(meta.get("user_task") or "unknown")
    return canonical_row(
        source="agentdojo-recorded-runs",
        source_id=source_id,
        tier="C_local_recorded",
        kind="action_authorization_guard",
        text=json.dumps(state, ensure_ascii=False, sort_keys=True),
        label=(corrected_label != 0),
        group_keys={
            "participant": f"{suite}:{user_task}",
            "template_cluster": str(meta.get("injection_task") or "benign"),
            "suite": suite,
        },
        supported_tasks=["action_authorization_guard"],
        metadata={
            "label_raw": corrected_label,
            "legacy_label": legacy_label,
            "view": view,
            "file": meta.get("file"),
            "suite": suite,
            "user_task": user_task,
            "injection_task": meta.get("injection_task"),
            "function": state["proposed_tool_call"]["function"],
            "label_rule": "corrected_v2_reconstructed_from_derived_state",
        },
    )


def _iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_llmail_raw(path):
    """Stream LLMail raw JSONL rows through the canonical converter."""
    for row in _iter_jsonl(path):
        yield convert_llmail_raw(row)


def iter_satml_chats(path):
    """Stream SaTML JSONL chats through the direct-injection converter."""
    for row in _iter_jsonl(path):
        yield convert_satml_chat(row)


def assign_group_splits(rows: Iterable[Mapping], seed: str = SPLIT_SEED) -> dict[str, list[dict]]:
    """Assign connected participant/template/text components to one split.

    Rows are unioned when they share a source-scoped participant, template
    cluster, or normalized text. Hashing each key independently would leak
    chained groups across splits; connected components prevent that leak.
    """
    materialized = [dict(row) for row in rows]
    parent = list(range(len(materialized)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_seen: dict[tuple[str, str, str], int] = {}
    for idx, row in enumerate(materialized):
        groups = row.get("group_keys") or {}
        tokens = [("text", "normalized", normalized_text_key(str(row["text"])))]
        for field in ("participant", "template_cluster"):
            if not str(groups.get(field, "")).strip():
                raise ValueError(f"row lacks group_keys.{field}")
            tokens.append((str(row["source"]), field, str(groups[field])))
        for token in tokens:
            if token in first_seen:
                union(idx, first_seen[token])
            else:
                first_seen[token] = idx

    components: dict[int, list[int]] = {}
    for idx in range(len(materialized)):
        components.setdefault(find(idx), []).append(idx)

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for indexes in components.values():
        representative = min(str(materialized[i]["row_id"]) for i in indexes)
        split = stable_group_split(representative, seed=seed)
        for i in indexes:
            materialized[i]["split"] = split
            splits[split].append(materialized[i])
    return splits


def assert_no_group_leakage(rows_by_split: Mapping[str, Iterable[Mapping]]) -> None:
    """Fail if participant, template cluster, or normalized text crosses splits."""
    seen: dict[tuple[str, str, str], str] = {}
    text_seen: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            text_key = normalized_text_key(str(row["text"]))
            if text_key in text_seen and text_seen[text_key] != split:
                raise ValueError(f"normalized text crosses {text_seen[text_key]} and {split}: {text_key}")
            text_seen.setdefault(text_key, split)
            groups = row.get("group_keys") or {}
            for field in ("participant", "template_cluster"):
                if not str(groups.get(field, "")).strip():
                    raise ValueError(f"row in {split} lacks group_keys.{field}")
                key = (str(row["source"]), field, str(groups[field]))
                if key in seen and seen[key] != split:
                    raise ValueError(f"group crosses {seen[key]} and {split}: {key}")
                seen.setdefault(key, split)

