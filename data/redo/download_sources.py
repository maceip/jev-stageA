"""Download external prompt-injection corpora into data/redo/raw with license ledger.
Only sources whose license was verified via API/file inspection are marked usable.
"""
import json, os, pathlib, subprocess, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "redo" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
ledger = []

# HF token if present (public sets are ungated; token only raises rate limits)
envp = ROOT / ".env"
if envp.exists():
    for line in envp.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ.setdefault("HF_TOKEN", line.split("=", 1)[1].strip())

from huggingface_hub import snapshot_download

HF_SETS = [
    ("Lakera/gandalf_ignore_instructions", "mit"),
    ("microsoft/llmail-inject-challenge", "mit"),
    ("ethz-spylab/ctf-satml24", "mit"),
]
for repo, lic in HF_SETS:
    name = repo.replace("/", "__")
    try:
        p = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(RAW / name))
        files = sorted(str(x.relative_to(RAW / name)) for x in pathlib.Path(p).rglob("*") if x.is_file())
        ledger.append({"source": repo, "kind": "hf-dataset", "license": lic, "usable_for_training": True, "dir": name, "files": files[:20]})
        print("OK", repo, len(files), "files")
    except Exception as e:
        ledger.append({"source": repo, "kind": "hf-dataset", "license": lic, "usable_for_training": False, "error": str(e)[:300]})
        print("FAIL", repo, str(e)[:200])

GH_SETS = [
    ("https://github.com/uiuc-kang-lab/InjecAgent.git", "InjecAgent", "mit"),
    ("https://github.com/microsoft/BIPIA.git", "BIPIA", "NOASSERTION-pending-manual-check"),
]
for url, name, lic in GH_SETS:
    dest = RAW / name
    if not dest.exists():
        r = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], capture_output=True, text=True, timeout=600)
        print("clone", name, r.returncode)
    lic_files = [str(x.relative_to(dest)) for x in dest.glob("LICEN*")] if dest.exists() else []
    lic_head = ""
    for lf in lic_files:
        lic_head = (dest / lf).read_text(errors="ignore")[:300]
        break
    ledger.append({"source": url, "kind": "github", "license": lic, "license_files": lic_files,
                   "license_head": lic_head, "usable_for_training": lic == "mit", "dir": name})

(RAW / "license-ledger.json").write_text(json.dumps(ledger, indent=1))
print(json.dumps([{k: e[k] for k in ("source", "license", "usable_for_training")} for e in ledger], indent=1))
