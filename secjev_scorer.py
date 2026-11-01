"""Batch SecJev scorer: loads a checkpoint once and scores Kev System One requests."""
import json, os, pathlib, sys, threading, time
import torch

WORK = pathlib.Path(os.environ["SECJEV_WORK"]).resolve()
sys.path.insert(0, str(WORK / "source/kev"))
from kev.api import SystemOneRequest, to_record, to_answers  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.model import rows_of  # noqa: E402

MAX_TOKENS = 8192


class SecJev:
    def __init__(self, model_dir: str, device: str = "cuda"):
        tf32 = os.environ.get("SECJEV_TF32") == "1"  # off = authors' reference numerics
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        ckpt = Checkpoint(model_dir)
        ckpt.meta.base = str(WORK / "base")
        ckpt.meta.base_revision = None
        self.tok, self.model = ckpt.load(device, LoadOptions(dtype=torch.float32, merge=False, attn="sdpa"))
        self.device = device
        self._lock = threading.Lock()  # Triton autotuning in fla kernels is not thread-safe

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
