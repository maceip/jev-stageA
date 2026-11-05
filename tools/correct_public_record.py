"""Apply the visible public-record correction to the three published redo repos.

FAIL-CLOSED. Requires:
  * TUESDAY_PUBLICATION_APPROVED=1 (explicit owner decision), and
  * a verified frozen chain manifest (G3-G7 complete), and
  * --apply (default is a dry run that only prints what would change).

No repository is modified when this script is merely imported or run with defaults.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tuesday_gates import canonical_secret, require_frozen_chain  # noqa: E402

REPOS = (
    "macmacmacmac/SecJev-4B-Spot2-Redo",
    "macmacmacmac/SecJev-4B-Spot1-Redo",
    "macmacmacmac/SecJev-4B-Joint-Redo",
)

NOTICE = """\n\n---\n\n## Correction notice (2026-11-04)\n\nThese weights are a **1-epoch research artifact** trained on non-frozen peer bench sets.\nTraining-time dev metrics for injection rows were computed with an incompatible score\nstatistic and are **withdrawn**. No frozen-test, adaptive, or replication result supports\nthese weights. They must not be used as evidence of a comparison against hosted\nJEV/Ultrafast models. A corrected multi-epoch frozen-split run supersedes this artifact;\nsee the project repository. The backbone was not fine-tuned: only the LoRA adapter and\nthe decision head were trained.\n"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out-dir", default="docs/redo-v2/corrected-cards")
    a = ap.parse_args()

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        (out / (repo.split("/")[-1] + ".md")).write_text(NOTICE.strip() + "\n")
    print("Prepared correction cards for", len(REPOS), "repos in", out)

    if not a.apply:
        print("DRY RUN: no repository modified.")
        return
    if os.environ.get("TUESDAY_PUBLICATION_APPROVED") != "1":
        raise SystemExit("BLOCKED: set TUESDAY_PUBLICATION_APPROVED=1 after the owner decides posture.")
    require_frozen_chain()
    token = canonical_secret("HF_TOKEN")
    if not token:
        raise SystemExit("BLOCKED: HF_TOKEN unresolved.")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for repo in REPOS:
        card = api.hf_hub_download(repo_id=repo, filename="README.md", repo_type="model")
        text = pathlib.Path(card).read_text()
        if "Correction notice (2026-11-04)" not in text:
            pathlib.Path(card).write_text(text.rstrip() + NOTICE)
            api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                            repo_id=repo, repo_type="model",
                            commit_message="Visible correction: 1-epoch non-frozen artifact; metrics withdrawn")
            print("CORRECTED", repo)
        else:
            print("ALREADY CORRECTED", repo)


if __name__ == "__main__":
    main()
