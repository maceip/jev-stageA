"""Batch SecJev scorer: loads a checkpoint once and scores Kev System One requests.

Two entry points:
  SecJev.score(request)        one request, one forward pass (unchanged API)
  SecJev.score_batch(requests)  a list of requests, batched into as few forward passes
                                as the batch limits allow; falls back to a loop of
                                SecJev.score() when the underlying Kev model cannot
                                batch (the fallback is logged).

Import safety: this module used to import torch and read SECJEV_WORK at import time, so
it could only be imported inside the scoring environment. It is now importable without
torch: the torch and kev imports are attempted and their absence is remembered, and the
SecJev constructor raises a clear error when the scoring environment is missing. When
torch and SECJEV_WORK *are* present the behaviour is exactly as before -- the kev
package is imported from $SECJEV_WORK/source/kev at import time, so `secjev_scorer.to_record`
and friends stay available at module level.

Batching: the Kev decision model exposes `forward_batch(encs)`, which runs the whole
batch through the backbone in one pass (right-padded rows, block-causal mask). The
records are grouped so that the padded batch stays inside a token budget, because the
mask is B x L x L and grows quadratically with the longest record.
"""
import json, os, pathlib, sys, threading, time
from typing import Iterator, List, Sequence

try:  # torch is only needed to *run* the model, not to import this module
    import torch
except Exception:  # pragma: no cover - depends on the environment
    torch = None

_WORK = os.environ.get("SECJEV_WORK")
WORK = pathlib.Path(_WORK).resolve() if _WORK else None

#: the kev package lives next to the checkpoint; import it the same way it always was
if WORK is not None and torch is not None:
    sys.path.insert(0, str(WORK / "source/kev"))
    from kev.api import SystemOneRequest, to_record, to_answers  # noqa: E402
    from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
    from kev.model import rows_of  # noqa: E402
    KEV_AVAILABLE = True
else:
    KEV_AVAILABLE = False
    SystemOneRequest = to_record = to_answers = Checkpoint = LoadOptions = rows_of = None

MAX_TOKENS = 8192
_GPU_LOCK = threading.Lock()  # module-level lock across all SecJev instances on the same GPU


def default_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

#: batch limits: at most this many records per forward pass, and at most this many
#: padded tokens (batch size x longest record) so the B x L x L attention mask stays
#: within the GPU budget. Both can be overridden from the environment.
BATCH_SIZE = int(os.environ.get("SECJEV_BATCH_SIZE", "4"))
BATCH_TOKENS = int(os.environ.get("SECJEV_BATCH_TOKENS", "16384"))


def chunked(items, n) -> Iterator[List]:
    """Yield lists of at most n items from items. Used by the batch path and the fallback."""
    items = list(items)
    n = int(n)
    if n <= 0:
        raise ValueError("chunk size must be a positive integer")
    for start in range(0, len(items), n):
        yield items[start:start + n]


def _log(message: str) -> None:
    print(f"[secjev_scorer] {message}", file=sys.stderr, flush=True)


class SecJev:
    def __init__(self, model_dir: str, device: str | None = None, batch_size: int = BATCH_SIZE,
                 batch_tokens: int = BATCH_TOKENS):
        if torch is None:
            raise RuntimeError("SecJev needs torch; install the scoring environment "
                               "(see RUNBOOK.md) before constructing it")
        if not KEV_AVAILABLE:
            raise RuntimeError("SecJev needs the kev package and SECJEV_WORK to be set; "
                               f"SECJEV_WORK={os.environ.get('SECJEV_WORK')!r}")
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
        self.batch_size = max(1, int(batch_size))
        self.batch_tokens = max(1, int(batch_tokens))
        self._lock = _GPU_LOCK  # shared lock prevents concurrent Triton autotuning / multi-model OOM

    # ------------------------------------------------------------------ single request

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

    # ------------------------------------------------------------------ batched requests

    def score_batch(self, requests: Sequence[dict], batch_size: int = None) -> List[dict]:
        """Score a list of requests, batching the forward passes.

        Returns one {question_id: {candidate: probability}} dict per request, in the order
        the requests were given. A request that does not fit the model's window raises
        ValueError (the same message the single-request path raises), so the caller's
        history-shrinking loop keeps working.

        If the underlying Kev model cannot batch, the batch is scored with a loop of
        self.score() instead and the fallback is logged.
        """
        requests = list(requests)
        if not requests:
            return []
        limit = self.batch_size if batch_size is None else max(1, int(batch_size))
        results: List[dict] = []
        for chunk in chunked(requests, limit):
            results.extend(self._score_chunk(chunk))
        return results

    def _score_chunk(self, chunk: List[dict]) -> List[dict]:
        """Score one chunk. Encoding happens per request (CPU), the forward pass is shared."""
        chunk_results: List[dict] = []
        encoded = []
        for request in chunk:
            req = SystemOneRequest.model_validate(request)
            record, metadata = to_record(req)
            enc = self.model.encode(self.tok, record, max_state=MAX_TOKENS, max_branch=MAX_TOKENS, strict=True)
            state, _, branches = rows_of(enc)
            if any(len(state) + len(b["ids"]) > MAX_TOKENS for b in branches):
                raise ValueError("state + question exceeds 8192 tokens")
            encoded.append((enc, metadata))
        for group in self._group_by_length(encoded):
            results = self._forward_group(group)
            if results is None:
                # the Kev model refused to batch: fall back to one request at a time
                return [self.score(request) for request in chunk]
            chunk_results.extend(results)
        return chunk_results

    def _group_by_length(self, encoded):
        """Split the encoded records into forward passes that respect the token budget."""
        groups = []
        current = []
        current_len = 0
        for item in encoded:
            length = max(len(item[0]["ids"]), 1)
            proposed = max(current_len, length) * (len(current) + 1)
            if current and (len(current) >= self.batch_size or proposed > self.batch_tokens):
                groups.append(current)
                current, current_len = [], 0
                proposed = length
            current.append(item)
            current_len = max(current_len, length)
        if current:
            groups.append(current)
        return groups

    def _forward_group(self, group):
        """One forward pass over a group of encoded records. None if the model cannot batch."""
        encs = [enc for enc, _ in group]
        metas = [metadata for _, metadata in group]
        try:
            with self._lock, torch.inference_mode():
                logits = self.model.forward_batch(encs)
        except Exception as exc:  # the model API may not support batching
            _log(f"batched forward failed ({type(exc).__name__}: {exc}); "
                 f"falling back to per-request scoring for {len(group)} request(s)")
            return None
        probs = [[torch.softmax(z.float(), -1).cpu().tolist() for z in record] for record in logits]
        return [{m["id"]: dict(zip(m["keys"], p)) for m, p in zip(meta, ps)} for meta, ps in zip(metas, probs)]
