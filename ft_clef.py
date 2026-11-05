"""Faithful fine-tuning of Clef-Flash (JointSchemaHead on 9B LiteRT Metal GPU hidden states).

Follows the exact train/dev/test protocol of ft_action.py (SecJev-4B-Action-Monitor-C1):
  - Train split: workspace + travel suites only (85% of user tasks via split_dedup.py SHA1 split)
  - Dev split:   workspace + travel suites (held-out 15% of user tasks via split_dedup.py SHA1 split)
  - Test split:  banking + slack suites (held-out domains, never used for training or checkpoint selection)

Trains JointSchemaHead only; the LiteRT backbone stays FROZEN. Trainable parameter count is
measured from the state dict at runtime (121,762,820 in the v1 run), never taken from a docstring.
end-to-end on the full [1, T, 4096] sequence hidden states produced by the 9B Clef-Flash LiteRT Metal GPU
backbone, jointly on Action Authorization Guard (content view) and Tool Output Filter injection replay.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from score import decision_points, secjev_tool_request  # noqa: E402
from fastjev_guard import resolve_eval_stamp  # noqa: E402

TRAIN_SUITES = {"workspace", "travel"}
# Strict by default: overlong rows are skipped and RECORDED, never silently truncated.
ALLOW_TRUNCATION = False
# --cache-only: hidden states were pre-encoded on CUDA; never touch LiteRT graphs.
CACHE_ONLY = False
TEST_SUITES = {"banking", "slack"}


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _is_dev_task(suite: str, user_task: str) -> bool:
    return int(_sha1(suite + user_task)[:4], 16) % 100 < 15


def _norm_inj_text(s: str) -> str:
    s = re.sub(r"\\\s*\n\s*\\", " ", s or "")
    return " ".join(s.replace("\\n", " ").split())


def _output_has_injection(ep: dict, output_text: str) -> bool:
    norm_out = _norm_inj_text(output_text)
    if not norm_out:
        return False
    for v in (ep.get("injections") or {}).values():
        nv = _norm_inj_text(v)
        if nv and nv in norm_out:
            return True
    return False


def build_datasets(out_dir: Path, seed: int = 20261101) -> dict[str, list[dict]]:
    """Assemble deduplicated train (workspace+travel), dev (workspace+travel held-out tasks),
    and test (banking+slack held-out suites) datasets combining Action Authorization Guard
    (content view) and Tool Output Filter injection detection."""
    rng = random.Random(seed)
    c_train = [json.loads(l) for l in open(ROOT / "ft/ft_c_train.jsonl")]
    c_dev = [json.loads(l) for l in open(ROOT / "ft/ft_c_dev.jsonl")]
    c_test = [json.loads(l) for l in open(ROOT / "ft/ft_c_test.jsonl")]

    m_train = [json.loads(l) for l in open(ROOT / "ft/ft_train.jsonl")]
    m_dev = [json.loads(l) for l in open(ROOT / "ft/ft_dev.jsonl")]

    def _dedup_action(c_rows: list[dict], m_pos_rows: list[dict], max_neg: int, max_m_pos: int) -> list[dict]:
        pos_c = [dict(r, _kind="action_authorization_guard") for r in c_rows if r["questions"]["authorization"]["label"] != 0]
        neg_c = [dict(r, _kind="action_authorization_guard") for r in c_rows if r["questions"]["authorization"]["label"] == 0]
        # Deduplicate negatives by (suite, user_task, proposed_tool_call) so one user_task repeated across
        # 14 injection_tasks does not dominate the epoch with identical read calls
        by_call = {}
        for r in neg_c:
            k = (
                r["_meta"]["suite"],
                r["_meta"]["user_task"],
                json.dumps(r["state"]["proposed_tool_call"], sort_keys=True),
            )
            # Prefer rows that actually have non-empty untrusted_tool_outputs_seen
            if k not in by_call or (
                r["state"].get("untrusted_tool_outputs_seen")
                and not by_call[k]["state"].get("untrusted_tool_outputs_seen")
            ):
                by_call[k] = r
        unique_negs = list(by_call.values())
        rng.shuffle(unique_negs)
        selected_negs = unique_negs[:max_neg]

        # Add distinct positive rows from ft_train / ft_dev (from gptoss20b / qwen30b runs)
        seen_pos_calls = {
            (r["_meta"]["suite"], r["_meta"]["user_task"], json.dumps(r["state"]["proposed_tool_call"], sort_keys=True))
            for r in pos_c
        }
        extra_pos = []
        for r in m_pos_rows:
            if r["questions"]["authorization"]["label"] == 0:
                continue
            k = (
                r["_meta"]["suite"],
                r["_meta"]["user_task"],
                json.dumps(r["state"]["proposed_tool_call"], sort_keys=True),
            )
            if k not in seen_pos_calls:
                seen_pos_calls.add(k)
                extra_pos.append(dict(r, _kind="action_authorization_guard"))
        rng.shuffle(extra_pos)
        return pos_c + extra_pos[:max_m_pos] + selected_negs

    # Build Tool Output Filter questions from recorded trajectories
    traj_files = [
        ROOT / "run3/benign.jsonl",
        ROOT / "run3/important_instructions.jsonl",
        ROOT / "run3/ood_bland_note.jsonl",
        ROOT / "run3/mini/benign.jsonl",
        ROOT / "run3/mini/important_instructions.jsonl",
        ROOT / "run3/mini/ood_bland_note.jsonl",
    ]
    tool_filter_by_split: dict[str, dict[str, dict]] = {"train": {}, "dev": {}, "test": {}}
    tool_filter_labs: dict[str, set[str]] = collections.defaultdict(set)
    for tf in traj_files:
        if not tf.exists():
            continue
        for line in open(tf):
            ep = json.loads(line)
            if ep.get("error") or ep["suite"] not in (TRAIN_SUITES | TEST_SUITES):
                continue
            sp = (
                ("dev" if _is_dev_task(ep["suite"], ep["user_task"]) else "train")
                if ep["suite"] in TRAIN_SUITES
                else "test"
            )
            for kind, _i, _k, p in decision_points(ep["messages"]):
                if kind != "tool":
                    continue
                out = p["output"]
                if ep.get("injection_task") is None:
                    label = "false"
                elif _output_has_injection(ep, out):
                    label = "true"
                else:
                    label = "false"
                req = secjev_tool_request(dict(p, output=out[:8000]))
                kh = _sha1(json.dumps(req["state"], sort_keys=True))
                tool_filter_labs[kh].add(label)
                q = dict(req["questions"]["injection"], label=label)
                row = {
                    "state": req["state"],
                    "questions": {"injection": q},
                    "_kind": "tool_output_filter",
                    "_meta": {
                        "suite": ep["suite"],
                        "file": str(tf.relative_to(ROOT)),
                        "user_task": ep["user_task"],
                        "injection_task": ep["injection_task"],
                    },
                }
                tool_filter_by_split[sp].setdefault(kh, row)

    def _select_tool_filter(sp: str, max_pos: int, max_neg: int) -> list[dict]:
        clean = [r for kh, r in tool_filter_by_split[sp].items() if len(tool_filter_labs[kh]) == 1]
        pos = [r for r in clean if r["questions"]["injection"]["label"] == "true"]
        neg = [r for r in clean if r["questions"]["injection"]["label"] == "false"]
        # Deduplicate negatives by (suite, user_task, tool_call.function)
        by_fn = {}
        for r in neg:
            k = (r["_meta"]["suite"], r["_meta"]["user_task"], r["state"]["tool_call"]["function"])
            by_fn.setdefault(k, r)
        u_neg = list(by_fn.values())
        rng.shuffle(pos)
        rng.shuffle(u_neg)
        return pos[:max_pos] + u_neg[:max_neg]

    train_s2 = _dedup_action(c_train, m_train, max_neg=160, max_m_pos=120)
    train_s1 = _select_tool_filter("train", max_pos=120, max_neg=120)
    train_rows = train_s2 + train_s1

    dev_s2 = _dedup_action(c_dev, m_dev, max_neg=45, max_m_pos=35)
    dev_s1 = _select_tool_filter("dev", max_pos=35, max_neg=35)
    dev_rows = dev_s2 + dev_s1

    test_s2 = _dedup_action(c_test, [], max_neg=60, max_m_pos=0)
    # Keep a balanced representative slice of test_s2 positives (up to 60 across banking+slack)
    test_s2_pos = [r for r in test_s2 if r["questions"]["authorization"]["label"] != 0]
    test_s2_neg = [r for r in test_s2 if r["questions"]["authorization"]["label"] == 0]
    rng.shuffle(test_s2_pos)
    test_s2 = test_s2_pos[:60] + test_s2_neg[:60]
    test_s1 = _select_tool_filter("test", max_pos=50, max_neg=50)
    test_rows = test_s2 + test_s1

    splits = {"train": train_rows, "dev": dev_rows, "test": test_rows}
    for sp, rows in splits.items():
        p = out_dir / f"clef_{sp}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return splits


def load_clef_components(litert_dir: Path, device: torch.device,
                         accelerator: str = "cpu", threads: int = 8):
    """Load ClefLiteRT (portable: cpu/XNNPACK or gpu delegate) and the PyTorch JointSchemaHead."""
    host_parent = str(litert_dir / "host")
    if host_parent not in sys.path:
        sys.path.insert(0, host_parent)
    from clef_litert.errors import RequestTooLong, SchemaTooLong
    from clef_litert.runtime import ClefLiteRT

    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", str(litert_dir / "joint_schema_model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    clef = ClefLiteRT.from_dir(litert_dir, accelerator=accelerator, threads=threads)
    clef.host.max_length = max(clef.graphs.lengths) + 1
    host = clef.host
    runners = clef.graphs.row_graph

    head_cfg = json.loads((litert_dir / "head/joint_head_config.json").read_text())
    head = mod.JointSchemaHead(**head_cfg)
    head.load_state_dict(load_file(str(litert_dir / "head/joint_head.safetensors")), strict=True)
    head = head.to(device=device, dtype=torch.float32)

    lm_head_np = host.lm_head
    return host, runners, head, mod, lm_head_np, (RequestTooLong, SchemaTooLong)


def load_clef_head_only(litert_dir: Path, device: torch.device):
    """CUDA pre-encode path: head + tokenizer host, ZERO LiteRT graph dependencies.

    Used with --cache-only. Hidden states come from tools/clef_gpu_encode.py, which
    runs the pinned PyTorch clef-flash backbone on CUDA. This is the concrete
    disproof of the claim that the CLEF harness must stay on the Mac.
    """
    host_parent = str(litert_dir / "host")
    if host_parent not in sys.path:
        sys.path.insert(0, host_parent)
    from clef_litert.errors import RequestTooLong, SchemaTooLong
    from clef_litert.host import ClefHost
    from clef_litert.positions import BUCKETS
    from tokenizers import Tokenizer

    spec = importlib.util.spec_from_file_location(
        "clef_joint_schema_model_cuda", str(litert_dir / "joint_schema_model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    class _TableStub:
        shape = (248320, 4096)

        def rows(self, ids):
            raise RuntimeError("CACHE_ONLY: lm_head rows must come from the pre-encoded cache")

    host = ClefHost(Tokenizer.from_file(str(litert_dir / "tokenizer/tokenizer.json")),
                    _TableStub(), _TableStub(), object(), buckets=BUCKETS)
    host.max_length = max(BUCKETS) + 1
    head_cfg = json.loads((litert_dir / "head/joint_head_config.json").read_text())
    head = mod.JointSchemaHead(**head_cfg)
    head.load_state_dict(load_file(str(litert_dir / "head/joint_head.safetensors")), strict=True)
    head = head.to(device=device, dtype=torch.float32)
    return host, None, head, mod, None, (RequestTooLong, SchemaTooLong)


def encode_and_extract(
    row: dict,
    host,
    runners,
    mod,
    RequestTooLong,
    cache_dir: Path,
):
    """Encode a labelled request and obtain its [T, 4096] backbone hidden states (using disk cache across epochs)."""
    qid, q = next(iter(row["questions"].items()))
    label = str(q["label"])
    clean_q = {kk: vv for kk, vv in q.items() if kk != "label"}
    req = {"model": "clef-flash", "state": row["state"], "questions": {qid: clean_q}}

    key = _sha1(json.dumps(req, sort_keys=True, ensure_ascii=False))
    cache_file = cache_dir / f"{key}.npz"
    if CACHE_ONLY and not cache_file.exists():
        with (cache_dir / "cache_misses.jsonl").open("a") as _f:
            _f.write(json.dumps({"key": key, "reason": "missing_preencoded_cache"}) + "\n")
        return None

    # Fit within 4096 tokens if needed
    cur_req = req
    while True:
        try:
            enc = host.encode(cur_req)
            break
        except RequestTooLong:
            if not ALLOW_TRUNCATION:
                with (cache_dir / "skipped_overlong.jsonl").open("a") as _f:
                    _f.write(json.dumps({"key": key, "reason": "RequestTooLong",
                                         "policy": "strict", "ts": time.time()}) + "\n")
                return None
            st = dict(cur_req["state"])
            if "untrusted_tool_response" in st and len(st["untrusted_tool_response"]) > 1000:
                st["untrusted_tool_response"] = st["untrusted_tool_response"][: len(st["untrusted_tool_response"]) // 2]
            elif st.get("untrusted_tool_outputs_seen"):
                st["untrusted_tool_outputs_seen"] = st["untrusted_tool_outputs_seen"][1:]
                if st.get("previous_tool_calls"):
                    st["previous_tool_calls"] = st["previous_tool_calls"][1:]
            else:
                return None
            cur_req = dict(cur_req, state=st)

    rec = enc.record
    t_len = rec.tokens
    q_meta = rec.questions[0]
    opt_ids = tuple(str(x) for x in q_meta.option_ids)
    tgt_idx = opt_ids.index(label)

    py_q = mod.EncodedQuestion(
        question_id=q_meta.question_id,
        question_type=q_meta.type_id,
        question_span=q_meta.question_span,
        option_spans=tuple(q_meta.option_spans),
        option_ids=opt_ids,
    )
    py_rec = mod.EncodedRecord(
        input_ids=tuple(int(x) for x in rec.ids),
        questions=(py_q,),
        record_id=key[:12],
        media=None,
    )

    if cache_file.exists():
        if CACHE_ONLY:
            # In cache-only mode, don't load hidden states into RAM.
            # Store the cache path so forward_single_head can load on-demand.
            data = np.load(cache_file)
            lex_rows = data["lexical"].astype(np.float32)
            seq_hidden = None  # loaded on-demand from cache_file
        else:
            data = np.load(cache_file)
            seq_hidden = data["hidden"].astype(np.float32)
            lex_rows = data["lexical"].astype(np.float32)
    else:
        row_in = host.row_inputs(enc)
        hidden_full = runners(enc.length, row_in)
        if hidden_full.ndim == 3:
            hidden_full = hidden_full[0]
        seq_hidden = np.asarray(hidden_full[:t_len], dtype=np.float32)
        # Pre-extract mean lexical vectors per option span from lm_head so we don't hold a 151k x 4096 table in GPU RAM
        ids_arr = np.asarray(rec.ids, dtype=np.int64)
        lex_list = [host.lm_head.rows(ids_arr[s:e]).mean(axis=0) for s, e in q_meta.option_spans]
        lex_rows = np.stack(lex_list, axis=0).astype(np.float32)
        np.savez_compressed(cache_file, hidden=seq_hidden.astype(np.float16), lexical=lex_rows.astype(np.float16))

    return {
        "key": key,
        "kind": row.get("_kind", "action_authorization_guard"),
        "suite": row["_meta"]["suite"],
        "hidden": seq_hidden,       # [T, 4096] float32 or None if cache-only (loaded on-demand)
        "cache_file": str(cache_file) if CACHE_ONLY else None,
        "lexical": lex_rows,        # [num_opts, 4096] float32 (small, ~16 KB per item)
        "record": py_rec,
        "opt_ids": opt_ids,
        "tgt": tgt_idx,
        "tokens": t_len,
    }


def forward_single_head(head, item: dict, device: torch.device) -> torch.Tensor:
    """Run JointSchemaHead on a single item's full [1, T, 4096] sequence hidden states using pre-extracted lexical vectors."""
    import math as _math

    hidden = item["hidden"]
    if hidden is None:
        # Cache-only mode: load from disk on-demand
        data = np.load(item["cache_file"])
        hidden = data["hidden"].astype(np.float32)
    hidden_states = torch.from_numpy(hidden).unsqueeze(0).to(device=device, dtype=torch.float32)
    lexical = torch.from_numpy(item["lexical"]).to(device=device, dtype=torch.float32)
    record = item["record"]
    q = record.questions[0]

    normalized_hidden = head.hidden_norm(hidden_states)
    sequence_hidden = normalized_hidden[0]  # [T, 4096]
    memory = head.memory_projection(sequence_hidden).unsqueeze(0)  # [1, T, width]
    global_vector = sequence_hidden[-1]
    question_vectors = head._mean_span(sequence_hidden, q.question_span).unsqueeze(0)  # [1, width_in]
    type_ids = torch.tensor([q.question_type], device=device)

    context_vectors = torch.stack([head._mean_span(sequence_hidden, span) for span in q.option_spans])
    option_queries = (
        head.option_context_projection(context_vectors)
        + head.option_lexical_projection(lexical)
        + head.option_question_projection(question_vectors[0]).unsqueeze(0)
    ).unsqueeze(0)

    routed_options = option_queries
    for layer in head.evidence_layers:
        routed_options = layer(routed_options, memory)
    routed = routed_options[0]  # [num_opts, width]

    base_fields = head.question_projection(question_vectors)  # [1, width]
    field0 = base_fields[0]
    routing_weights = torch.softmax(torch.matmul(routed, field0) / _math.sqrt(routed.shape[-1]), dim=0)
    option_summary = torch.sum(routing_weights.unsqueeze(-1) * routed, dim=0, keepdim=True)

    fields = (
        base_fields
        + head.option_summary_norm(option_summary)
        + head.global_projection(global_vector).unsqueeze(0)
        + head.type_embedding(type_ids)
    ).unsqueeze(0)
    for layer in head.layers:
        fields = layer(fields, memory)
    field = head.field_norm(fields[0])[0]

    anchor = F.normalize(question_vectors[0] + global_vector, dim=-1)
    lexical_anchor = F.normalize(lexical, dim=-1)
    prior_scale = head.prior_logit_scale.clamp(max=_math.log(100.0)).exp()
    prior = prior_scale * torch.matmul(lexical_anchor, anchor)

    options = head.option_norm(routed)
    repeated_field = field.unsqueeze(0).expand_as(options)
    cosine = F.cosine_similarity(repeated_field, options, dim=-1)
    features = torch.cat(
        [repeated_field, options, repeated_field * options, torch.abs(repeated_field - options)],
        dim=-1,
    )
    residual = head.residual_scorer(features).squeeze(-1)
    joint_scale = head.joint_logit_scale.clamp(max=_math.log(100.0)).exp()
    joint = joint_scale * cosine + residual
    logits = prior + torch.sigmoid(head.residual_gate) * joint
    return logits


def compute_metrics(head, items: list[dict], device: torch.device, tag: str, epoch: int, step: int,
                    rows_out: list[dict] | None = None) -> dict:
    head.eval()
    s1_scores, s2_scores = [], []
    losses = []
    with torch.no_grad():
        for it in items:
            z = forward_single_head(head, it, device)
            loss = float(F.cross_entropy(z.unsqueeze(0), torch.tensor([it["tgt"]], device=device)).item())
            losses.append(loss)
            probs = dict(zip(it["opt_ids"], torch.softmax(z.float(), dim=-1).tolist()))
            if it["kind"] == "tool_output_filter":
                score = probs.get("true", 0.0)
                is_pos = it["opt_ids"][it["tgt"]] == "true"
                s1_scores.append((score, is_pos))
            else:
                score = probs.get("2", 0.0) + 0.5 * probs.get("1", 0.0)
                is_pos = it["opt_ids"][it["tgt"]] != "0"
                s2_scores.append((score, is_pos))
            if rows_out is not None:
                rows_out.append({"kind": it["kind"], "score": round(float(score), 6), "is_pos": bool(is_pos)})
    head.train()

    def _summarize(pairs, thrs):
        pos = [s for s, p in pairs if p]
        neg = [s for s, p in pairs if not p]
        auc = (
            sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / max(1, len(pos) * len(neg))
            if pos and neg
            else 0.0
        )
        out = {"n": len(pairs), "pos": len(pos), "neg": len(neg), "auc": round(auc, 4)}
        for thr in thrs:
            out[f"tpr@{thr}"] = round(sum(s >= thr for s in pos) / max(1, len(pos)), 4)
            out[f"fpr@{thr}"] = round(sum(s >= thr for s in neg) / max(1, len(neg)), 4)
        return out

    return {
        "event": "eval",
        "split": tag,
        "epoch": epoch,
        "step": step,
        "loss": round(sum(losses) / max(1, len(losses)), 4),
        "tool_output_filter": _summarize(s1_scores, (0.5, 0.9, 0.99)),
        "action_authorization_guard": _summarize(s2_scores, (0.35, 0.5, 0.9)),
    }


def save_checkpoint(head, base_dir: Path, dest_dir: Path, meta_info: dict):
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("tokenizer", "tables", "host", "joint_schema_model.py", "contract.json"):
            src = base_dir / sub
            dst = dest_dir / sub
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.exists():
                shutil.copy2(src, dst)
        for tf in base_dir.glob("*.tflite"):
            dst_tf = dest_dir / tf.name
            if not dst_tf.exists():
                shutil.copy2(tf.resolve(), dst_tf)  # self-contained: real file, not a symlink
    head_dir = dest_dir / "head"
    head_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_dir / "head/joint_head_config.json", head_dir / "joint_head_config.json")
    state_dict = {k: v.detach().cpu().to(torch.bfloat16).contiguous()
                  for k, v in head.state_dict().items()}  # deployment contract is BF16
    save_file(state_dict, str(head_dir / "joint_head.safetensors"))
    (dest_dir / "provenance_ft.json").write_text(json.dumps(meta_info, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: cuda, else mps, else cpu)")
    ap.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"],
                    help="LiteRT delegate; gpu is Metal on macOS, cpu/XNNPACK is portable")
    ap.add_argument("--threads", type=int, default=8, help="LiteRT CPU delegate threads")
    ap.add_argument("--allow-truncation", action="store_true",
                    help="DISCOURAGED: silently halving overlong tool text was the v1 defect")
    ap.add_argument("--cache-only", action="store_true",
                    help="use pre-encoded CUDA hidden states; no LiteRT graphs needed")
    ap.add_argument("--data-dir", default="",
                    help="frozen corpus dir holding clef_{train,dev,test}.jsonl; bypasses the capped builder")
    ap.add_argument("--init", default=str(ROOT / "checkpoints/clef-flash-LiteRT"))
    ap.add_argument("--out", default=str(ROOT / "ft/clef_ft_run"))
    ap.add_argument("--ckpt_out", default=str(ROOT / "checkpoints/clef-flash-LiteRT-ft"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pos_weight", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--eval-interval", type=int, default=0,
                    help="dev eval every N steps (0 = epoch-boundary only, default for full-corpus scale)")
    a = ap.parse_args()
    global ALLOW_TRUNCATION, CACHE_ONLY
    ALLOW_TRUNCATION = a.allow_truncation
    CACHE_ONLY = a.cache_only

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    log_file = open(log_path, "a")

    def log_event(obj: dict):
        obj["ts"] = round(time.time(), 2)
        line = json.dumps(obj)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    device = torch.device(
        a.device or ("cuda" if torch.cuda.is_available()
                     else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                     else "cpu"))
    if a.data_dir:
        splits = {
            sp: [json.loads(l) for l in open(Path(a.data_dir) / f"clef_{sp}.jsonl")]
            for sp in ("train", "dev", "test")
        }
        log_event({"event": "frozen_data_loaded", "data_dir": a.data_dir,
                   "counts": {k: len(v) for k, v in splits.items()}, "capped": False})
    else:
        splits = build_datasets(out_dir, seed=a.seed)
    log_event(
        {
            "event": "start",
            "device": str(device),
            "train_raw": len(splits["train"]),
            "dev_raw": len(splits["dev"]),
            "test_raw": len(splits["test"]),
            "lr": a.lr,
            "epochs": a.epochs,
            "batch": a.batch,
            "pos_weight": a.pos_weight,
        }
    )

    litert_dir = Path(a.init)
    if a.cache_only:
        host, runners, head, mod, _lm_head_np, RequestTooLong = load_clef_head_only(litert_dir, device)
        log_event({"event": "cache_only_mode", "litert_graphs": 0,
                   "note": "hidden states pre-encoded with pinned PyTorch clef-flash on CUDA"})
    else:
        host, runners, head, mod, _lm_head_np, RequestTooLong = load_clef_components(
            litert_dir, device, accelerator=a.accelerator, threads=a.threads)
    params = [p for p in head.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    state_n = sum(v.numel() for v in head.state_dict().values())
    log_event({"event": "model_loaded", "trainable_params": n_params,
               "state_dict_params": state_n, "source": "runtime_state_dict"})

    t0 = time.time()
    # 1. Encode Dev split first and run pre-fine-tuning baseline evaluation (Epoch 0)
    dev_items = []
    for idx, r in enumerate(splits["dev"], 1):
        item = encode_and_extract(r, host, runners, mod, RequestTooLong, cache_dir)
        if item is not None:
            dev_items.append(item)
        if idx % 20 == 0 or idx == len(splits["dev"]):
            log_event(
                {
                    "event": "prep_dev",
                    "done": idx,
                    "of": len(splits["dev"]),
                    "elapsed_min": round((time.time() - t0) / 60, 2),
                }
            )

    if CACHE_ONLY and not any(cache_dir.glob("*.npz")):
        raise RuntimeError(
            "CACHE_ONLY: no pre-encoded *.npz cache files found in the --out/cache dir. "
            "Run tools/clef_gpu_encode.py first. Refusing to train.")
    if CACHE_ONLY and not dev_items:
        raise RuntimeError(
            "CACHE_ONLY produced zero dev items: the pre-encoded cache is empty or "
            "keys do not match. Refusing to report zero-filled metrics.")
    eval0 = compute_metrics(head, dev_items, device, "dev", epoch=0, step=0)
    eval0["elapsed_min"] = round((time.time() - t0) / 60, 2)
    log_event(eval0)

    # 2. Training loop: stream-encode in Epoch 1 (training immediately on each batch as hidden states arrive)
    # and reuse disk-cached [T, 4096] hidden states in Epochs 2..N
    steps_per_epoch = math.ceil(len(splits["train"]) / a.batch)
    total_steps = steps_per_epoch * a.epochs
    warm_steps = max(1, round(0.08 * total_steps))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm_steps
        if s < warm_steps
        else 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (s - warm_steps) / max(1, total_steps - warm_steps))),
    )

    best_score = -1.0
    best_epoch = None
    best_step = None
    step = 0
    encoded_train_items: list[dict] = []

    for epoch in range(1, a.epochs + 1):
        if epoch == 1:
            raw_order = list(splits["train"])
            random.shuffle(raw_order)
            for i in range(0, len(raw_order), a.batch):
                raw_chunk = raw_order[i : i + a.batch]
                batch_items = []
                for r in raw_chunk:
                    it = encode_and_extract(r, host, runners, mod, RequestTooLong, cache_dir)
                    if it is not None:
                        batch_items.append(it)
                        encoded_train_items.append(it)
                if not batch_items:
                    continue
                head.train()
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for it in batch_items:
                    z = forward_single_head(head, it, device)
                    is_pos = (
                        (it["opt_ids"][it["tgt"]] == "true")
                        if it["kind"] == "tool_output_filter"
                        else (it["opt_ids"][it["tgt"]] != "0")
                    )
                    w = a.pos_weight if is_pos else 1.0
                    loss = w * F.cross_entropy(z.unsqueeze(0), torch.tensor([it["tgt"]], device=device))
                    (loss / len(batch_items)).backward()
                    batch_loss += float(loss.item())
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 200 == 0 or step == total_steps:
                    log_event(
                        {
                            "event": "step",
                            "epoch": epoch,
                            "step": step,
                            "of": total_steps,
                            "encoded_train": len(encoded_train_items),
                            "train_total": len(raw_order),
                            "loss": round(batch_loss / len(batch_items), 4),
                            "lr": round(sched.get_last_lr()[0], 7),
                            "elapsed_min": round((time.time() - t0) / 60, 2),
                        }
                    )
                if a.eval_interval > 0 and step % a.eval_interval == 0:
                    ev = compute_metrics(head, dev_items, device, "dev", epoch=epoch, step=step)
                    ev["elapsed_min"] = round((time.time() - t0) / 60, 2)
                    log_event(ev)
        else:
            items_order = list(encoded_train_items)
            random.shuffle(items_order)
            for i in range(0, len(items_order), a.batch):
                batch_items = items_order[i : i + a.batch]
                head.train()
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for it in batch_items:
                    z = forward_single_head(head, it, device)
                    is_pos = (
                        (it["opt_ids"][it["tgt"]] == "true")
                        if it["kind"] == "tool_output_filter"
                        else (it["opt_ids"][it["tgt"]] != "0")
                    )
                    w = a.pos_weight if is_pos else 1.0
                    loss = w * F.cross_entropy(z.unsqueeze(0), torch.tensor([it["tgt"]], device=device))
                    (loss / len(batch_items)).backward()
                    batch_loss += float(loss.item())
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 200 == 0 or i + a.batch >= len(items_order):
                    log_event(
                        {
                            "event": "step",
                            "epoch": epoch,
                            "step": step,
                            "of": total_steps,
                            "encoded_train": len(encoded_train_items),
                            "train_total": len(encoded_train_items),
                            "loss": round(batch_loss / len(batch_items), 4),
                            "lr": round(sched.get_last_lr()[0], 7),
                            "elapsed_min": round((time.time() - t0) / 60, 2),
                        }
                    )

        ev = compute_metrics(head, dev_items, device, "dev", epoch=epoch, step=step)
        ev["elapsed_min"] = round((time.time() - t0) / 60, 2)
        log_event(ev)
        comb_auc = 0.5 * (ev["tool_output_filter"]["auc"] + ev["action_authorization_guard"]["auc"])
        save_checkpoint(
            head,
            litert_dir,
             out_dir / f"epoch{epoch}",
            {"epoch": epoch, "step": step, "dev_eval": ev, "lr": a.lr, "pos_weight": a.pos_weight, "seed": a.seed},
        )
        if comb_auc >= best_score:
            best_score = comb_auc
            best_epoch = epoch
            best_step = step
            save_checkpoint(
                head,
                litert_dir,
                Path(a.ckpt_out),
                {"epoch": epoch, "step": step, "dev_eval": ev, "lr": a.lr, "pos_weight": a.pos_weight, "seed": a.seed},
            )
            log_event({"event": "saved_best", "epoch": epoch, "step": step, "comb_auc": round(comb_auc, 4)})

    # 3. Held-out Test evaluation on banking + slack (never used during training or checkpoint selection)
    best_head_sd = load_file(str(Path(a.ckpt_out) / "head/joint_head.safetensors"))
    head.load_state_dict(best_head_sd, strict=True)
    test_items = []
    for idx, r in enumerate(splits["test"], 1):
        it = encode_and_extract(r, host, runners, mod, RequestTooLong, cache_dir)
        if it is not None:
            test_items.append(it)
        if idx % 20 == 0 or idx == len(splits["test"]):
            log_event(
                {
                    "event": "prep_test",
                    "done": idx,
                    "of": len(splits["test"]),
                    "elapsed_min": round((time.time() - t0) / 60, 2),
                }
            )
    stamp = resolve_eval_stamp(best_epoch, best_step, requested_epochs=a.epochs, final_step=step)
    dev_rows: list[dict] = []
    dev_best = compute_metrics(head, dev_items, device, "dev_best", rows_out=dev_rows, **stamp)
    (out_dir / "dev_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev_rows))
    test_rows: list[dict] = []
    test_ev = compute_metrics(head, test_items, device, "test_banking_slack", rows_out=test_rows, **stamp)
    test_ev["elapsed_min"] = round((time.time() - t0) / 60, 2)
    test_ev["checkpoint"] = {"path": str(Path(a.ckpt_out)), **stamp}
    log_event(test_ev)
    (out_dir / "test_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test_rows))
    (out_dir / "final_results.json").write_text(
        json.dumps({"dev_baseline": eval0, "dev_best": dev_best, "test_banking_slack": test_ev}, indent=2)
    )
    log_event({"event": "train_done", "elapsed_min": round((time.time() - t0) / 60, 2)})


if __name__ == "__main__":
    main()
