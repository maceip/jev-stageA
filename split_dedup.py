"""Concatenate parts, drop states with conflicting labels, dedupe, split train/dev (15% of user tasks held out). Usage: split_dedup.py PARTS_DIR OUT_PREFIX"""
import collections, glob, hashlib, json, sys
d, pre = sys.argv[1], sys.argv[2]
h = lambda x: hashlib.sha1(x.encode()).hexdigest()
for split in ("train", "test"):
    labs = collections.defaultdict(set); first = {}
    for f in sorted(glob.glob(f"{d}/*_{split}.jsonl")):
        for l in open(f):
            r = json.loads(l); k = h(json.dumps(r["state"], sort_keys=True))
            labs[k].add(r["questions"]["authorization"]["label"]); first.setdefault(k, r)
    rows = [r for k, r in first.items() if len(labs[k]) == 1]
    sel = lambda r: int(h(r["_meta"]["suite"] + r["_meta"]["user_task"])[:4], 16) % 100 < 15
    groups = {"train": [r for r in rows if not sel(r)], "dev": [r for r in rows if sel(r)]} if split == "train" else {"test": rows}
    for name, rs in groups.items():
        open(f"{pre}_{name}.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rs)
        print(name, len(rs), collections.Counter(r["questions"]["authorization"]["label"] for r in rs))
