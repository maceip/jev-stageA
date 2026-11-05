"""Score all CLEF heads (base + FT configs) on bench devcal/eval rows with ONE backbone encode pass.

Rationale: CLEF fine-tuning changes only the JointSchemaHead; backbone hidden states are
head-independent and cached by ft_clef_bench.encode_and_extract (SHA1 of request).
CLEF eval scope (predeclared, disclosed): 200 pos + 200 neg per track drawn from the frozen
eval files (seed 4242), identical rows for every CLEF arm; devcal sets used in full for
threshold calibration. Other arms were scored on the full 782/800 draws.

Usage:
  .venv/bin/python data/redo/bench_clef_heads.py encode     # resumable encode pass
  .venv/bin/python data/redo/bench_clef_heads.py score      # all heads -> ft/bench_results/clef_*.json
"""
import json, pathlib, random, resource, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import redo_eval
import ft_clef_bench as FC

BENCH = ROOT / "data" / "redo" / "bench"
OUT = ROOT / "ft" / "bench_results"
CACHE = ROOT / "ft" / "bench_clef_cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "checkpoints" / "clef-flash-LiteRT"

def load(name):
    return [json.loads(l) for l in open(BENCH / name)]

def lab(r, qid):
    return r["questions"][qid]["label"] not in (0, "false", False)

def clef_eval_draw(rows, qid, npos=200, nneg=200, seed=4242):
    pos = [r for r in rows if lab(r, qid)]; neg = [r for r in rows if not lab(r, qid)]
    rnd = random.Random(seed); rnd.shuffle(pos); rnd.shuffle(neg)
    sel = pos[:npos] + neg[:nneg]; rnd.shuffle(sel); return sel

SETS = {}
def all_sets():
    if SETS: return SETS
    SETS["dev_spot1"] = load("devcal_spot1.jsonl")
    SETS["dev_spot2"] = load("devcal_spot2.jsonl")
    SETS["eval_spot1"] = clef_eval_draw(load("eval_spot1.jsonl"), "injection")
    SETS["eval_spot2"] = clef_eval_draw(load("eval_spot2.jsonl"), "authorization")
    return SETS

def get_components():
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    host, runners, mod, RequestTooLong, head0 = FC.load_clef_components(BASE, device)
    return host, runners, mod, RequestTooLong, device

def encode_all():
    host, runners, mod, RTL, device = get_components()
    for name, rows in all_sets().items():
        items, t0 = [], time.time()
        for i, r in enumerate(rows, 1):
            it = FC.encode_and_extract(r, host, runners, mod, RTL, CACHE)
            if it is not None: items.append(it["key"])
            if i % 25 == 0:
                print(f"{name} {i}/{len(rows)} {round((time.time()-t0)/60,1)}m", flush=True)
        (CACHE / f"keys_{name}.json").write_text(json.dumps({"keys": items, "encode_seconds": round(time.time()-t0, 1)}))
        print(name, "encoded", len(items), flush=True)

def load_head(ckpt_dir, mod, device):
    import torch
    from safetensors.torch import load_file
    cfg = json.loads((pathlib.Path(ckpt_dir) / "head/joint_head_config.json").read_text())
    head = mod.JointSchemaHead(**cfg)
    sd = load_file(str(pathlib.Path(ckpt_dir) / "head/joint_head.safetensors"))
    head.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return head.to(device).eval()

def items_for(name, host, runners, mod, RTL):
    rows = all_sets()[name]
    out = []
    for r in rows:
        it = FC.encode_and_extract(r, host, runners, mod, RTL, CACHE)
        if it is not None: out.append((it, r))
    return out

def score_items(head, pairs, qid, device):
    import torch
    out = []
    with torch.no_grad():
        for it, r in pairs:
            logits = FC.forward_single_head(head, it, device)
            p = torch.softmax(logits.float(), -1).cpu().tolist()
            d = dict(zip(it["opt_ids"], p))
            s = float(d.get("true", 0.0)) if qid == "injection" else float(d.get("2", 0.0) + 0.5 * d.get("1", 0.0))
            out.append({"score": s, "label": lab(r, qid), "group": (r.get("_meta") or {}).get("suite", "?")})
    return out

def track_metrics(rows, thr):
    m = redo_eval.evaluate_scores(rows, thr)
    fp = sum(1 for r in rows if not r["label"] and r["score"] >= thr)
    tp = sum(1 for r in rows if r["label"] and r["score"] >= thr)
    tn = sum(1 for r in rows if not r["label"] and r["score"] < thr)
    fn = sum(1 for r in rows if r["label"] and r["score"] < thr)
    m.update({"fp": fp, "tp": tp, "tn": tn, "fn": fn, "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4)})
    return m

ARMS = [
    ("clef_base", BASE, 0.0),
    ("clef_spot1", ROOT / "checkpoints/redo-clef-spot1", None),
    ("clef_spot2", ROOT / "checkpoints/redo-clef-spot2", None),
    ("clef_joint", ROOT / "checkpoints/redo-clef-joint", None),
]

def score_all(ft_minutes_map):
    host, runners, mod, RTL, device = get_components()
    sets = {n: items_for(n, host, runners, mod, RTL) for n in all_sets()}
    enc_secs = {}
    for n in sets:
        p = CACHE / f"keys_{n}.json"
        enc_secs[n] = json.loads(p.read_text())["encode_seconds"] if p.exists() else 0.0
    for arm, ckpt, ftm in ARMS:
        if not pathlib.Path(ckpt).exists():
            print("SKIP missing", arm, ckpt, flush=True); continue
        head = load_head(ckpt, mod, device)
        result = {"arm": arm, "kind": "local-clef-heads", "ckpt": str(ckpt),
                  "ft_minutes": (ft_minutes_map or {}).get(arm, ftm or 0.0), "tracks": {},
                  "note": "CLEF eval scope: 200pos+200neg/track from frozen eval draws (seed 4242); backbone encoded once, heads scored from cached features"}
        for track, qid, devn, evn in [("tool_output_filter", "injection", "dev_spot1", "eval_spot1"),
                                       ("action_authz_guard", "authorization", "dev_spot2", "eval_spot2")]:
            t0 = time.time()
            dev_s = score_items(head, sets[devn], qid, device)
            ev_s = score_items(head, sets[evn], qid, device)
            thr = redo_eval.calibrate_threshold(dev_s, 0.05)
            m = track_metrics(ev_s, thr)
            m.update({"dev_threshold": thr,
                      "run_seconds_dev_plus_test": round(enc_secs.get(devn, 0) + enc_secs.get(evn, 0) + (time.time() - t0), 1)})
            result["tracks"][track] = m
            (OUT / f"{arm}.scores.{track}.json").write_text(json.dumps(ev_s))
            print(arm, track, json.dumps(m), flush=True)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_gb"] = round(rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9, 2)
        (OUT / f"{arm}.json").write_text(json.dumps(result, indent=1))
        print("wrote", arm, flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "encode":
        encode_all()
    elif sys.argv[1] == "score":
        ftm = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        score_all(ftm)
