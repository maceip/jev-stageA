"""SecJev scorer with bounded exact-request reuse and ordered multi-request scoring.

Unique requests use the validated single-request Kev forward path. score_many deduplicates inputs;
it is not GPU minibatching, and its throughput must not be reported as measured GPU speedup.
"""
import os, pathlib, sys, threading

from score import CachedScorer

MAX_TOKENS = 8192


class SecJev:
    def __init__(self, model_dir: str, device: str = "cuda", cache_size: int = 4096):
        if not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        # Import only when a real model is requested; pure request/cache utilities need no GPU stack.
        import torch
        work = pathlib.Path(os.environ["SECJEV_WORK"]).resolve()
        sys.path.insert(0, str(work / "source/kev"))
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.model import rows_of

        tf32 = os.environ.get("SECJEV_TF32") == "1"  # off = authors' reference numerics
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        ckpt = Checkpoint(model_dir)
        ckpt.meta.base = str(work / "base")
        ckpt.meta.base_revision = None
        self.tok, self.model = ckpt.load(device, LoadOptions(dtype=torch.float32, merge=False, attn="sdpa"))
        self.device = device
        self._torch, self._request_type = torch, SystemOneRequest
        self._to_record, self._rows_of = to_record, rows_of
        self._lock = threading.Lock()  # Triton autotuning in fla kernels is not thread-safe
        self._scorer = CachedScorer(self._score_uncached, cache_size)

    def score(self, request: dict) -> dict:
        """Return {question_id: {candidate: probability}}; never cache model errors."""
        return self._scorer.score(request)

    def score_many(self, requests) -> list[dict]:
        """Score distinct exact requests once and restore input order (sequential unique forwards)."""
        return self._scorer.score_many(requests)

    def cache_info(self):
        return self._scorer.cache_info()

    def _score_uncached(self, request):
        req = self._request_type.model_validate(request)
        record, metadata = self._to_record(req)
        enc = self.model.encode(self.tok, record, max_state=MAX_TOKENS, max_branch=MAX_TOKENS, strict=True)
        state, _, branches = self._rows_of(enc)
        if any(len(state) + len(b["ids"]) > MAX_TOKENS for b in branches):
            raise ValueError("state + question exceeds 8192 tokens")
        with self._lock, self._torch.inference_mode():
            logits = self.model.forward(enc)
        probs = [self._torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
        if len(metadata) != len(probs) or any(len(m["keys"]) != len(p) for m, p in zip(metadata, probs)):
            raise ValueError("SecJev returned an incomplete question/candidate score set")
        return {m["id"]: dict(zip(m["keys"], p)) for m, p in zip(metadata, probs)}
