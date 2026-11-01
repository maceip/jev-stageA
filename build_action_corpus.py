"""Extract leak-free Action-Authorization training & calibration pairs from recorded AgentDojo archives.

Splits strictly by suite:
  fold_A (tune/train): workspace, travel
  fold_B (eval/test) : banking, slack
Guarantees zero user_task, tool_call schema, or environment overlap between train and test splits.
"""
import json, tarfile
from collections import Counter
from pathlib import Path
from score import decision_points, secjev_action_request

FOLD_A = {"workspace", "travel"}
FOLD_B = {"banking", "slack"}


def extract_archive(tgz_path: Path, model_tag: str):
    benign_fns = {}
    records = []
    with tarfile.open(tgz_path, "r:gz") as tf:
        for line in tf.extractfile(f"runs/{model_tag}/benign.jsonl"):
            ep = json.loads(line)
            if ep.get("error") is not None:
                continue
            key = (ep["suite"], ep["user_task"])
            bset = benign_fns.setdefault(key, set())
            for kind, i, k, p in decision_points(ep["messages"]):
                if kind == "action":
                    bset.add(p["call"]["function"])
                    if ep.get("utility"):
                        records.append({
                            "model": model_tag,
                            "suite": ep["suite"],
                            "fold": "A" if ep["suite"] in FOLD_A else "B",
                            "label": "0",  # Authorized
                            "source": "benign",
                            "request": secjev_action_request(p, True),
                        })
        for atk in ("important_instructions", "ood_bland_note", "ood_metadata_field",
                    "ood_quoted_reply", "ood_zero_width", "ood_llmail"):
            for line in tf.extractfile(f"runs/{model_tag}/{atk}.jsonl"):
                ep = json.loads(line)
                if ep.get("error") is not None or not ep.get("attack_success"):
                    continue
                bset = benign_fns.get((ep["suite"], ep["user_task"]), set())
                for kind, i, k, p in decision_points(ep["messages"]):
                    if kind == "action" and p["call"]["function"] not in bset:
                        records.append({
                            "model": model_tag,
                            "suite": ep["suite"],
                            "fold": "A" if ep["suite"] in FOLD_A else "B",
                            "label": "2",  # Unauthorized (hijacked call)
                            "source": atk,
                            "request": secjev_action_request(p, True),
                        })
    return records


def main():
    root = Path(__file__).resolve().parent
    all_recs = []
    for tag, fn in (("gptoss20b", "gptoss20b_stageA.tgz"), ("qwen30b", "qwen30b_stageA.tgz")):
        p = root / "data" / fn
        if p.exists():
            all_recs.extend(extract_archive(p, tag))
    counts = Counter((r["fold"], r["label"]) for r in all_recs)
    by_suite = Counter((r["suite"], r["label"]) for r in all_recs)
    summary = {
        "total_pairs": len(all_recs),
        "fold_A_workspace_travel": {"authorized_0": counts[("A", "0")], "unauthorized_2": counts[("A", "2")]},
        "fold_B_banking_slack": {"authorized_0": counts[("B", "0")], "unauthorized_2": counts[("B", "2")]},
        "by_suite": {f"{s}:label_{l}": c for (s, l), c in sorted(by_suite.items())},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
