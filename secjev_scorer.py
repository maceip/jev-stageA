"""Batch SecJev scorer: loads a checkpoint once and scores Kev System One requests."""
import json, os, pathlib, sys, threading, time
import torch

WORK = pathlib.Path(os.environ["SECJEV_WORK"]).resolve()
sys.path.insert(0, str(WORK / "source/kev"))
from kev.api import SystemOneRequest, to_record, to_answers  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.model import rows_of  # noqa: E402

MAX_TOKENS = 8192
_GPU_LOCK = threading.Lock()  # module-level lock across all SecJev instances on the same GPU


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SecJev:
    def __init__(self, model_dir: str, device: str | None = None):
        device = device or default_device()
        tf32 = os.environ.get("SECJEV_TF32") == "1"  # off = authors' reference numerics
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        ckpt = Checkpoint(model_dir)
        if device == "mps" and os.environ.get("KEV_BACKEND", "mlx") == "mlx":
            opts = LoadOptions(backend="mlx")
        else:
            ckpt.meta.base = str(WORK / "base")
            ckpt.meta.base_revision = None
            opts = LoadOptions(dtype=torch.float32, merge=False, attn="sdpa")
        self.tok, self.model = ckpt.load(device, opts)
        self.device = device
        self._lock = _GPU_LOCK  # shared lock prevents concurrent Triton autotuning / multi-model OOM
        # Stacked forward is opt-in: it changes kernel shapes, so it must be
        # validated against sequential scores before it is trusted for results.
        self.stack = os.environ.get("SECJEV_STACK") == "1"
        self.deduped = self.batched = self.fallbacks = 0

    def score(self, request: dict) -> dict:
        """Return {question_id: {candidate: probability}}. Raises if input exceeds the model's window."""
        req = SystemOneRequest.model_validate(request)
        record, metadata = to_record(req)
        enc = self.model.encode(self.tok, record, max_state=MAX_TOKENS, max_branch=MAX_TOKENS, strict=True)
        state, _, branches = rows_of(enc)
        if any(len(state) + len(b["ids"]) > MAX_TOKENS for b in branches):
            raise ValueError("state + question exceeds 8192 tokens")
        with self._lock, torch.inference_mode():
            logits = self.model.forward(enc)
        probs = [torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
        return {m["id"]: dict(zip(m["keys"], p)) for m, p in zip(metadata, probs)}

    def score_many(self, requests: list[dict]) -> list[dict]:
        """Score N requests with N' <= N model calls.

        Two speedups, both measurable and both safe to disable:

        1. Duplicate suppression. Score pipelines re-request identical payloads
           constantly (the same tool output re-scored for every attack set, the
           same action point re-scored when chunking retries). Requests are
           keyed by their serialized content, so each distinct payload is
           encoded and forwarded exactly once.
        2. Stacked forward. When `self.stack` is on, the distinct encodings are
           pushed through one `forward` call. SecJev's encoder emits padded
           rows, so stacking is attempted and, if the checkpoint or the kernel
           refuses it, the batch degrades to N' sequential forwards - never to a
           wrong number.

        The lock is taken once per batch rather than once per request, which is
        the part that matters when 48 threads share one GPU.
        """
        keys = [json.dumps(r, sort_keys=True, default=str) for r in requests]
        distinct: dict[str, dict] = {}
        for key, request in zip(keys, requests):
            distinct.setdefault(key, request)
        if len(distinct) < len(requests):
            self.deduped += len(requests) - len(distinct)
        resolved = self._stacked_forward(distinct) if (self.stack and len(distinct) > 1) else \
            {k: self.score(v) for k, v in distinct.items()}
        out = []
        for key in keys:
            value = resolved.get(key)
            out.append(value if value is not None else {"error": "batch item missing"})
        return out

    def _stacked_forward(self, distinct: dict[str, dict]) -> dict[str, dict]:
        """Try one GPU pass over a batch; fall back to sequential on any failure."""
        try:
            encoded, metas = [], []
            for request in distinct.values():
                req = SystemOneRequest.model_validate(request)
                record, metadata = to_record(req)
                enc = self.model.encode(self.tok, record, max_state=MAX_TOKENS,
                                        max_branch=MAX_TOKENS, strict=True)
                encoded.append(enc)
                metas.append(metadata)
            stacked = self._stack(encoded)
            with self._lock, torch.inference_mode():
                logits = self.model.forward(stacked)
            probs = [torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
            self.batched += len(encoded)
            return {key: {m["id"]: dict(zip(m["keys"], p)) for m, p in zip(meta, probs)}
                    for key, meta in zip(distinct, metas)}
        except Exception:
            self.fallbacks += 1
            return {key: self.score(request) for key, request in distinct.items()}

    @staticmethod
    def _stack(encoded):
        """Concatenate per-request encodings along the batch axis.

        SecJev encodings are dicts of tensors/lists; only entries whose first
        axis is the batch axis are stackable, and a shape mismatch anywhere
        aborts the batch so the caller falls back.
        """
        import torch as _torch
        first = encoded[0]
        if not isinstance(first, dict):
            raise TypeError("unsupported encoding type for stacking")
        out = {}
        for key in first:
            values = [e[key] for e in encoded]
            if all(isinstance(v, _torch.Tensor) and v.shape[1:] == values[0].shape[1:]
                   for v in values):
                out[key] = _torch.cat(values, dim=0)
            else:
                out[key] = values
        return out

    def speed_stats(self) -> dict:
        return {"deduped_requests": self.deduped, "batched_requests": self.batched,
                "batch_fallbacks": self.fallbacks}
