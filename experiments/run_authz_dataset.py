"""Run the action-authorization dataset builder and summarise the result.

Thin CLI around build_authz_dataset: it runs the builder, prints a compact JSON summary
and writes a Markdown report to <jev-stageA>/results/authz_dataset_report.md.

    python experiments/run_authz_dataset.py [--tar T ...] [--out DIR] [--folds 2]
                                            [--max-per-suite N] [--seed S]

Stdlib only.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_authz_dataset as builder  # noqa: E402

REPORT = ROOT / "results" / "authz_dataset_report.md"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_report(folds_doc, stats, path):
    leakage = folds_doc["leakage_report"]
    cross = folds_doc["train_test_leakage"]
    counts = folds_doc["counts"]
    suites = sorted(stats["per_suite"])

    parts = ["# Action-authorization dataset",
             "",
             f"Inputs: `{', '.join(stats['inputs'])}`  ",
             f"Output: `{stats['out_dir']}`  ",
             f"Folds: {stats['folds']} (suite -> fold: "
             f"{', '.join(f'{s}={f}' for s, f in sorted(folds_doc['fold_of_suite'].items()))})",
             "",
             "## Counts",
             "",
             markdown_table(["split", "examples", "authorized", "unauthorized", "unknown"], [
                 ["train", counts["train"]] + [stats["label_counts_train"].get(k, 0)
                                               for k in builder.LABELS],
                 ["test", counts["test"]] + [stats["label_counts_test"].get(k, 0)
                                             for k in builder.LABELS],
                 ["unknown (held out)", counts["unknown"]] + [0, 0, counts["unknown"]],
             ]),
             "",
             "## Per suite",
             "",
             markdown_table(["suite", "fold", "all", "authorized", "unauthorized", "unknown", "tools"], [
                 [s, stats["per_suite"][s]["fold"],
                  sum(stats["per_suite"][s]["all"].values()),
                  stats["per_suite"][s]["all"].get("authorized", 0),
                  stats["per_suite"][s]["all"].get("unauthorized", 0),
                  stats["per_suite"][s]["all"].get("unknown", 0),
                  stats["suite_tool_counts"][s]]
                 for s in suites]),
             "",
             "## Per fold",
             "",
             markdown_table(["fold", "suites", "labels"], [
                 [f, ", ".join(v["suites"]),
                  ", ".join(f"{k}={n}" for k, n in sorted(v["labels"].items()))]
                 for f, v in sorted(stats["per_fold"].items())]),
             "",
             "## Label rules",
             "",
             markdown_table(["rule", "all", "train", "test"], [
                 [rule, stats["rule_counts"].get(rule, 0),
                  stats["rule_counts_train"].get(rule, 0),
                  stats["rule_counts_test"].get(rule, 0)]
                 for rule in sorted(stats["rule_counts"])]),
             "",
             "## Cross-suite tool overlap (Jaccard)",
             "",
             markdown_table(["suite"] + suites,
                            [[a] + [stats["tool_overlap_matrix"][a][b] for b in suites]
                             for a in suites]),
             "",
             f"Maximum pairwise overlap: **{leakage['max_pairwise_jaccard']}** "
             f"({' vs '.join(leakage['max_pairwise_jaccard_suites']) or 'n/a'}).",
             "",
             markdown_table(["suite A", "suite B", "shared tools", "jaccard"], [
                 [p["suites"][0], p["suites"][1], p["shared_count"], p["jaccard"]]
                 for p in leakage["pairs"]]),
             "",
             f"Train/test overlap: train = {', '.join(cross['train_suites'])}; "
             f"test = {', '.join(cross['test_suites'])}; "
             f"maximum cross-split Jaccard = **{cross['max_cross_split_jaccard']}**"
             + (f" ({' vs '.join(cross['max_cross_split_jaccard_suites'])})"
                if cross["max_cross_split_jaccard"] > 0 else "")
             + (f"; shared tools: {', '.join(cross['shared_functions'])}"
                if cross["shared_functions"] else "; no shared tools"),
             "",
             ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tar", action="append", default=None, help="tarball with runs/ and scores/")
    parser.add_argument("--out", default=str(builder.DEFAULT_OUT), help="dataset output directory")
    parser.add_argument("--folds", type=int, default=builder.DEFAULT_FOLDS, help="number of suite folds")
    parser.add_argument("--max-per-suite", type=int, default=0, help="cap examples per suite (0 = no cap)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the per-suite cap")
    parser.add_argument("--report", default=str(REPORT), help="Markdown report path")
    args = parser.parse_args(argv)

    argv_for_builder = ["--out", args.out, "--folds", str(args.folds),
                        "--max-per-suite", str(args.max_per_suite), "--seed", str(args.seed)]
    for tar in (args.tar or []):
        argv_for_builder += ["--tar", tar]
    builder.main(argv_for_builder, quiet=True)

    out_dir = Path(args.out)
    folds_doc = json.loads((out_dir / "authz_folds.json").read_text(encoding="utf-8"))
    stats = json.loads((out_dir / "authz_stats.json").read_text(encoding="utf-8"))
    report_path = write_report(folds_doc, stats, Path(args.report))
    # results/ is gitignored in this repository, so also drop a copy next
    # to the dataset metadata: the report is a tracked deliverable while
    # the bulk *.jsonl corpus stays regenerable and untracked.
    mirror = out_dir / "authz_dataset_report.md"
    write_report(folds_doc, stats, mirror)

    summary = {
        "examples": stats["label_counts"],
        "train": folds_doc["counts"]["train"],
        "test": folds_doc["counts"]["test"],
        "unknown": folds_doc["counts"]["unknown"],
        "per_suite": {s: stats["per_suite"][s]["all"] for s in sorted(stats["per_suite"])},
        "per_fold": {f: v["labels"] for f, v in sorted(stats["per_fold"].items())},
        "rules": stats["rule_counts"],
        "max_cross_suite_tool_overlap": folds_doc["leakage_report"]["max_pairwise_jaccard"],
        "max_cross_suite_tool_overlap_suites":
            folds_doc["leakage_report"]["max_pairwise_jaccard_suites"],
        "max_train_test_tool_overlap": folds_doc["train_test_leakage"]["max_cross_split_jaccard"],
        "report": str(report_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
