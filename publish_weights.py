"""Publish both JEV defense checkpoints to Hugging Face as complete, runnable checkpoint directories.

Assembles, from local sources:
  1. <org>/SecJev-4B                    unmodified upstream SecJev-4B v1.2.0 mirror (Tool Output Filter)
  2. <org>/SecJev-4B-Action-Monitor-C1  upstream directory with the ft/ckpt C1 adapter/head/config
                                        overlaid, a C1 provenance record, and upstream records moved
                                        under upstream-secjev-4b/ (Action Authorization Guard)
Each repository root is a Kev checkpoint directory, so the Kev loader accepts the Hub repo id directly.

Usage:
  HF_TOKEN=... python publish_weights.py --org MetaPublic --upstream ~/SecJev/SecJev-4B [--staging /tmp/hf-secjev]
  HF_TOKEN=... python publish_weights.py --org MetaPublic --upstream ~/SecJev/SecJev-4B --assemble-only
The Qwen3.5-4B-Base backbone is not re-hosted: bootstrap.py in each repository downloads the pinned
backbone and Kev source from the upstream distributors.
"""
import argparse, hashlib, json, os, pathlib, shutil

C1_HASHES = {
    "adapter_model.safetensors": "184fbde48a39ca6a8c3f227518581a24e7af2b478db1dac58e028e455284494c",
    "head.pt": "93ec77f5536d29dcd468b21ff519e5c2885ddbe0d685ff05f9ab50d1ce892a51",
}
UPSTREAM_RECORDS = ["checkpoint-selection.json", "epoch-3-test.json", "epoch-3-validation.json",
                    "export-validation.json", "security-training-config.json", "evaluation"]


def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def assemble(upstream, root, ft_ckpt, org):
    upstream, root = pathlib.Path(upstream).expanduser(), pathlib.Path(root)
    tool, action = root / "SecJev-4B", root / "SecJev-4B-Action-Monitor-C1"
    for dest in (tool, action):
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(upstream, dest, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    for name in ("adapter_model.safetensors", "head.pt", "adapter_config.json"):
        shutil.copy2(pathlib.Path(ft_ckpt) / name, action / name)
    (action / "upstream-provenance.json").write_bytes((action / "provenance.json").read_bytes())
    sub = action / "upstream-secjev-4b"
    sub.mkdir()
    for name in UPSTREAM_RECORDS:
        shutil.move(str(action / name), str(sub / name))
    for name, digest in C1_HASHES.items():
        assert sha256(action / name) == digest, f"C1 hash mismatch: {name}"
    cards = pathlib.Path(__file__).parent / "hf"
    shutil.copy2(cards / "SecJev-4B.README.md", tool / "README.md")
    shutil.copy2(cards / "SecJev-4B-Action-Monitor-C1.README.md", action / "README.md")
    shutil.copy2(cards / "SecJev-4B-Action-Monitor-C1.provenance.json", action / "provenance.json")
    return tool, action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="macmacmacmac")
    ap.add_argument("--upstream", required=True, help="local unmodified SecJev-4B v1.2.0 release directory")
    ap.add_argument("--ft-ckpt", default="ft/ckpt")
    ap.add_argument("--staging", default="/tmp/hf-secjev")
    ap.add_argument("--assemble-only", action="store_true")
    a = ap.parse_args()
    tool, action = assemble(a.upstream, a.staging, a.ft_ckpt, a.org)
    print(f"assembled {tool} and {action}; C1 hashes verified")
    if a.assemble_only:
        return
    token = os.environ.get("HF_TOKEN")
    if not token and pathlib.Path(".env").is_file():
        for line in pathlib.Path(".env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for local, repo in ((tool, f"{a.org}/SecJev-4B"), (action, f"{a.org}/SecJev-4B-Action-Monitor-C1")):
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(local), repo_id=repo, repo_type="model",
                          commit_message="Complete runnable SecJev checkpoint for jev-stageA JEV defense")
        print(f"uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
