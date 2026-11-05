"""Apply the Tuesday-assessment §5.1 fixes to the jev-stageA training/eval scripts.

Every replacement is asserted. Run with --check to see coverage without writing.
This script is idempotent: it records already-applied replacements as SKIP.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (file, old, new, label)
PATCHES: list[tuple[str, str, str, str]] = []


def add(files, old, new, label):
    for f in files:
        PATCHES.append((f, old, new, label))


CLEF3 = ["ft_clef.py", "ft_clef_redo.py", "ft_clef_bench.py"]
ACTION2 = ["ft_action.py", "ft_action_redo.py"]

# --- 1. Honest docstring / parameter description -------------------------------
add(CLEF3,
    "Trains JointSchemaHead (60.8M parameters: EvidenceRoutingLayers + TransformerDecoderLayers + residual_scorer)",
    "Trains JointSchemaHead only; the LiteRT backbone stays FROZEN. Trainable parameter count is\n"
    "measured from the state dict at runtime (121,762,820 in the v1 run), never taken from a docstring.",
    "docstring-param-count")

# --- 2. Portable LiteRT delegate + device -------------------------------------
add(CLEF3,
    "def load_clef_components(litert_dir: Path, device: torch.device):",
    "def load_clef_components(litert_dir: Path, device: torch.device,\n"
    "                         accelerator: str = \"cpu\", threads: int = 8):",
    "litert-signature")
add(CLEF3,
    '    """Load ClefLiteRT (ClefHost + Metal GPU row_graph) for the 9B backbone and PyTorch JointSchemaHead."""',
    '    """Load ClefLiteRT (portable: cpu/XNNPACK or gpu delegate) and the PyTorch JointSchemaHead."""',
    "litert-doc")
add(CLEF3,
    '    clef = ClefLiteRT.from_dir(litert_dir, accelerator="gpu", threads=8)',
    "    clef = ClefLiteRT.from_dir(litert_dir, accelerator=accelerator, threads=threads)",
    "litert-accelerator")

add(CLEF3,
    '    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")',
    '    device = torch.device(\n'
    '        a.device or ("cuda" if torch.cuda.is_available()\n'
    '                     else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()\n'
    '                     else "cpu"))',
    "cuda-device")

add(CLEF3,
    "    host, runners, head, mod, _lm_head_np, RequestTooLong = load_clef_components(litert_dir, device)",
    "    host, runners, head, mod, _lm_head_np, RequestTooLong = load_clef_components(\n"
    "        litert_dir, device, accelerator=a.accelerator, threads=a.threads)",
    "litert-call")

# --- 3. CLI: explicit device, delegate, frozen data, strict length -------------
add(CLEF3,
    "    ap = argparse.ArgumentParser()",
    "    ap = argparse.ArgumentParser()\n"
    '    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: cuda, else mps, else cpu)")\n'
    '    ap.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"],\n'
    '                    help="LiteRT delegate; gpu is Metal on macOS, cpu/XNNPACK is portable")\n'
    '    ap.add_argument("--threads", type=int, default=8, help="LiteRT CPU delegate threads")\n'
    '    ap.add_argument("--allow-truncation", action="store_true",\n'
    '                    help="DISCOURAGED: silently halving overlong tool text was the v1 defect")\n'
    '    ap.add_argument("--data-dir", default="",\n'
    '                    help="frozen corpus dir holding clef_{train,dev,test}.jsonl; bypasses the capped builder")',
    "cli-flags")

# global flag for strict length handling
add(CLEF3,
    "TRAIN_SUITES = {\"workspace\", \"travel\"}",
    "TRAIN_SUITES = {\"workspace\", \"travel\"}\n"
    "# Strict by default: overlong rows are skipped and RECORDED, never silently truncated.\n"
    "ALLOW_TRUNCATION = False",
    "strict-global")

add(CLEF3,
    "    ap.parse_args()",
    "    ap.parse_args()",  # placeholder, replaced below per-file via main-hook
    "noop")

# set the global from CLI right after parse
add(CLEF3,
    "    a = ap.parse_args()\n\n    random.seed(a.seed)",
    "    a = ap.parse_args()\n"
    "    global ALLOW_TRUNCATION\n"
    "    ALLOW_TRUNCATION = a.allow_truncation\n\n"
    "    random.seed(a.seed)",
    "strict-from-cli")

# --- 4. Frozen data bypass (removes the 569/500/1000-row caps) -----------------
add(CLEF3,
    "    splits = build_datasets(out_dir, seed=a.seed)",
    "    if a.data_dir:\n"
    "        splits = {\n"
    '            sp: [json.loads(l) for l in open(Path(a.data_dir) / f"clef_{sp}.jsonl")]\n'
    '            for sp in ("train", "dev", "test")\n'
    "        }\n"
    '        log_event({"event": "frozen_data_loaded", "data_dir": a.data_dir,\n'
    '                   "counts": {k: len(v) for k, v in splits.items()}, "capped": False})\n'
    "    else:\n"
    "        splits = build_datasets(out_dir, seed=a.seed)",
    "frozen-data-bypass")

# --- 5. Strict length: record skips, do not mutate the input -------------------
add(CLEF3,
    "        except RequestTooLong:\n            st = dict(cur_req[\"state\"])",
    "        except RequestTooLong:\n"
    "            if not ALLOW_TRUNCATION:\n"
    '                with (cache_dir / "skipped_overlong.jsonl").open("a") as _f:\n'
    '                    _f.write(json.dumps({"key": key, "reason": "RequestTooLong",\n'
    '                                         "policy": "strict", "ts": time.time()}) + "\\n")\n'
    "                return None\n"
    '            st = dict(cur_req["state"])',
    "strict-length")

# --- 6. Parameter count from the state dict -----------------------------------
add(CLEF3,
    '    log_event({"event": "model_loaded", "trainable_params": n_params})',
    '    state_n = sum(v.numel() for v in head.state_dict().values())\n'
    '    log_event({"event": "model_loaded", "trainable_params": n_params,\n'
    '               "state_dict_params": state_n, "source": "runtime_state_dict"})',
    "param-from-state")

# --- 7. Self-contained BF16 export -------------------------------------------
add(CLEF3,
    '        for sub in ("tokenizer", "tables", "host", "joint_schema_model.py"):',
    '        for sub in ("tokenizer", "tables", "host", "joint_schema_model.py", "contract.json"):',
    "export-contract")
add(CLEF3,
    "        for tf in base_dir.glob(\"*.tflite\"):\n"
    "            dst_tf = dest_dir / tf.name\n"
    "            if not dst_tf.exists():\n"
    "                os.symlink(tf.resolve(), dst_tf)",
    "        for tf in base_dir.glob(\"*.tflite\"):\n"
    "            dst_tf = dest_dir / tf.name\n"
    "            if not dst_tf.exists():\n"
    "                shutil.copy2(tf.resolve(), dst_tf)  # self-contained: real file, not a symlink",
    "export-graph")
# ft_clef_bench.py already casts to bf16 at line 355; only the other two need it.
add(["ft_clef.py", "ft_clef_redo.py"],
    "    state_dict = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}",
    "    state_dict = {k: v.detach().cpu().to(torch.bfloat16).contiguous()\n"
    "                  for k, v in head.state_dict().items()}  # deployment contract is BF16",
    "export-bf16")

# --- 8. ft_action: label-key-agnostic dev scoring ------------------------------
add(ACTION2,
    "MAX = 8192",
    "MAX = 8192\n\n\n"
    "def verdict_score(p, keys):\n"
    '    """Positive-class score for BOTH question types.\n\n'
    '    `noul` (injection) questions emit keys ["true","false"]; `score` (authorization)\n'
    '    questions emit ["0","1","2"]. The old p.get("2") statistic returned 0.0 for every\n'
    '    injection row, which is why spot1/joint training-time dev metrics were invalid.\n'
    '    """\n'
    '    if "true" in keys or "false" in keys:\n'
    '        return float(p.get("true", 0.0))\n'
    '    if "2" in keys:\n'
    '        return float(p.get("2", 0.0)) + 0.5 * float(p.get("1", 0.0))\n'
    '    return max([float(v) for k, v in p.items() if k != "0"] or [0.0])',
    "verdict-score")
add(["ft_action.py"],
    '                    P.append((p.get("2", 0.0) + 0.5 * p.get("1", 0.0), keys[tgt[0]]))',
    "                    P.append((verdict_score(p, keys), keys[tgt[0]]))",
    "verdict-call")
# ft_action_redo.py has a shallower indentation in its evaluate loop.
add(["ft_action_redo.py"],
    '                P.append((p.get("2", 0.0) + 0.5 * p.get("1", 0.0), keys[tgt[0]]))',
    "                P.append((verdict_score(p, keys), keys[tgt[0]]))",
    "verdict-call")
add(ACTION2,
    '        pos = [s for s, t in P if t != "0"]; neg = [s for s, t in P if t == "0"]',
    '        pos = [s for s, t in P if t not in ("0", "false")]\n'
    '        neg = [s for s, t in P if t in ("0", "false")]',
    "verdict-labels")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    by_file: dict[str, list] = {}
    for f, old, new, label in PATCHES:
        by_file.setdefault(f, []).append((old, new, label))

    total = applied = skipped = missing = 0
    for fname, reps in by_file.items():
        p = ROOT / fname
        if not p.exists():
            print(f"MISSING FILE {fname}")
            missing += len(reps)
            continue
        text = p.read_text()
        orig = text
        file_tags = []
        for old, new, label in reps:
            total += 1
            if old == new:
                continue
            if new in text and old not in text:
                skipped += 1
                file_tags.append(f"{label}=APPLIED_EARLIER")
            elif old in text:
                text = text.replace(old, new, 1)
                applied += 1
                file_tags.append(f"{label}=OK")
            else:
                missing += 1
                file_tags.append(f"{label}=NO_MATCH")
        if not a.check and text != orig:
            p.write_text(text)
        print(f"{fname:24s} " + " ".join(file_tags))
    print(f"\nTOTAL={total} APPLIED={applied} ALREADY={skipped} NO_MATCH={missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
