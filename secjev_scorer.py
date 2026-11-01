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
        self._cache = {}

    def _encode(self, request: dict):
        req = SystemOneRequest.model_validate(request)
        record, metadata = to_record(req)
        enc = self.model.encode(self.tok, record, max_state=MAX_TOKENS, max_branch=MAX_TOKENS, strict=True)
        state, _, branches = rows_of(enc)
        if any(len(state) + len(b["ids"]) > MAX_TOKENS for b in branches):
            raise ValueError("state + question exceeds 8192 tokens")
        return enc, metadata

    @staticmethod
    def _pack(logits, metadata):
        probs = [torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
        return {m["id"]: dict(zip(m["keys"], p)) for m, p in zip(metadata, probs)}

    def score(self, request: dict) -> dict:
        """Return {question_id: {candidate: probability}}. Identical requests hit an in-process cache."""
        key = json.dumps(request, sort_keys=True, default=str)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        enc, metadata = self._encode(request)
        with self._lock, torch.inference_mode():
            logits = self.model.forward(enc)
        out = self._pack(logits, metadata)
        if len(self._cache) >= 4096:
            self._cache.clear()
        self._cache[key] = out
        return out

    def score_batch(self, requests: list[dict]) -> list[dict]:
        """One lock acquisition and one forward_batch for many Kev requests.

        Repeated requests are served from the same cache as score(). Uncached rows share a single
        padded forward_batch (kev.model.Model.forward_batch) instead of a Python loop of batch-size-1 forwards.
        """
        if not requests:
            return []
        keys = [json.dumps(request, sort_keys=True, default=str) for request in requests]
        missing = []
        for key, request in zip(keys, requests):
            if key not in self._cache and key not in {k for k, _ in missing}:
                missing.append((key, request))
        if missing:
            prepared = [(key, *self._encode(request)) for key, request in missing]
            with self._lock, torch.inference_mode():
                batched = self.model.forward_batch([enc for _, enc, _ in prepared])
            for (key, _, metadata), logits in zip(prepared, batched):
                if len(self._cache) >= 4096:
                    self._cache.clear()
                self._cache[key] = self._pack(logits, metadata)
        return [self._cache[key] for key in keys]
