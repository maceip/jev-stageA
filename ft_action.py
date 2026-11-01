"""Fine-tune SecJev-4B (LoRA + pointer head) on action-authorization questions, with SecJev-Corpus replay.

One process per GPU. Usage:
  python ft_action.py --train train.jsonl --dev dev.jsonl --replay replay.jsonl --replay_frac 0.3 \
      --lr 3e-5 --epochs 2 --pos_weight 4 --out runs/x
Writes checkpoints loadable by secjev_scorer.SecJev (same layout as SecJev-4B) after each epoch.
"""
import argparse, copy, json, math, os, pathlib, random, shutil, sys, time
import torch

WORK = pathlib.Path(os.environ["SECJEV_WORK"]).resolve()
sys.path.insert(0, str(WORK / "source/kev"))
from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions, write_meta  # noqa: E402
from kev.model import rows_of  # noqa: E402

MAX = 8192


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if p else []


def encode(model, tok, req):
    """-> list of (enc, label_index) or None if over the window. Requests carry 'label' inside each question."""
    labels = {qid: q.get("label") for qid, q in req["questions"].items()}
    clean = {"state": req["state"], "questions": {k: {kk: vv for kk, vv in q.items() if kk != "label"}
                                                     for k, q in req["questions"].items()}}
    record, meta = to_record(SystemOneRequest.model_validate(clean))
    try:
        enc = model.encode(tok, record, max_state=MAX, max_branch=MAX, strict=True)
    except ValueError as e:
        if "exceeds" in str(e): return None
        raise
    s, _, br = rows_of(enc)
    if any(len(s) + len(b["ids"]) > MAX for b in br):
        return None
    tgt = []
    for m in meta:
        lab = labels[m["id"]]
        tgt.append([str(k) for k in m["keys"]].index(str(lab)) if lab is not None else None)
    return enc, tgt, [str(k) for k in meta[0]["keys"]], len(s) + max(len(b["ids"]) for b in br)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(pathlib.Path.home() / "SecJev/SecJev-4B"))
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--replay", default="")
    ap.add_argument("--replay_frac", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32, help="questions per optimizer step")
    ap.add_argument("--pos_weight", type=float, default=1.0, help="loss weight for label!=0 examples")
    ap.add_argument("--tok_budget", type=int, default=16384)
    ap.add_argument("--checkpointing", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    ck = Checkpoint(a.init)
    ck.meta.base = str(WORK / "base"); ck.meta.base_revision = None
    tok, model = ck.load("cuda", LoadOptions(dtype=torch.float32, merge=False, attn="sdpa", temperature=1.0))
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n) or n.startswith("head.")
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable", sum(p.numel() for p in params), flush=True)
    model.lm.config.use_cache = False
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.lm.enable_input_require_grads()

    def prep(rows, tag):
        res, skipped = [], 0
        for r in rows:
            e = encode(model, tok, r)
            if e is None: skipped += 1; continue
            res.append(e)
        print(tag, "encoded", len(res), "skipped_over_window", skipped, flush=True)
        return res

    train = prep(load_jsonl(a.train), "train")
    dev = prep(load_jsonl(a.dev), "dev")
    replay = prep(load_jsonl(a.replay), "replay") if a.replay_frac > 0 else []
    n_rep = int(len(train) * a.replay_frac / max(1e-9, 1 - a.replay_frac)) if replay else 0

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil((len(train) + n_rep) / a.batch)
    total = steps_per_epoch * a.epochs; warm = max(1, round(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    log = open(out / "log.jsonl", "a")

    def evaluate(epoch):
        model.eval(); P = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for enc, tgt, keys, _L in dev:
                z = model.forward(enc)[0].float()
                p = dict(zip(keys, torch.softmax(z, -1).tolist()))
                P.append((p.get("2", 0.0) + 0.5 * p.get("1", 0.0), keys[tgt[0]]))
        model.train()
        pos = [s for s, t in P if t != "0"]; neg = [s for s, t in P if t == "0"]
        auc = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / max(1, len(pos) * len(neg))
        rec = {"epoch": epoch, "dev_n": len(P), "dev_pos": len(pos), "auc": auc}
        for thr in (0.5, 0.9):
            rec[f"tpr@{thr}"] = sum(s >= thr for s in pos) / max(1, len(pos))
            rec[f"fpr@{thr}"] = sum(s >= thr for s in neg) / max(1, len(neg))
        print("EVAL", json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

    def save(epoch):
        dest = out / f"epoch{epoch}"
        shutil.copytree(a.init, dest)
        model.lm.peft_config["default"].base_model_name_or_path = "Qwen/Qwen3.5-4B-Base"
        model.lm.save_pretrained(dest)
        meta = copy.deepcopy(Checkpoint(a.init).meta)
        meta.head = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
        meta.temperature = 1.0
        meta.extra["action_ft"] = {"epoch": epoch, "lr": a.lr, "replay_frac": a.replay_frac, "pos_weight": a.pos_weight,
                                   "train": a.train, "seed": a.seed, "temperature": "uncalibrated"}
        write_meta(dest, meta)
        print("SAVED", dest, flush=True)

    evaluate(0)
    step = 0; t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        items = [(x, 1.0) for x in train] + [(x, 1.0) for x in random.sample(replay, min(n_rep, len(replay)))]
        random.shuffle(items)
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            nq = sum(sum(t is not None for t in x[1]) for x, _ in chunk)
            opt.zero_grad(set_to_none=True); tot = 0.0
            chunk.sort(key=lambda x: x[0][3])
            micros, cur = [], []
            for x in chunk:
                if cur and (len(cur) + 1) * x[0][3] > a.tok_budget: micros.append(cur); cur = []
                cur.append(x)
            micros.append(cur)
            for micro in micros:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    zss = model.forward_batch([x[0][0] for x in micro])
                loss = 0.0
                for zs, ((enc, tgt, _k, _L), _w) in zip(zss, micro):
                    for z, t in zip(zs, tgt):
                        if t is None: continue
                        w = a.pos_weight if (t != 0 and len(tgt) == 1 and len(z) == 3) else 1.0
                        loss = loss + w * torch.nn.functional.cross_entropy(z.float()[None], torch.tensor([t], device=z.device))
                if not torch.is_tensor(loss): continue
                (loss / nq).backward(); tot += float(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 10 == 0:
                rec = {"step": step, "of": total, "loss": tot / max(1, nq), "lr": sched.get_last_lr()[0], "min": (time.time() - t0) / 60}
                print("STEP", json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        save(epoch); evaluate(epoch)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
