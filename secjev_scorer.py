"""Batch SecJev & Clef scorer: loads a checkpoint once and scores Kev / Clef System One requests.

Supports:
  - SecJev / Kev checkpoints on Linux/NVIDIA (CUDA), Apple Silicon (PyTorch MPS or MLX), and CPU.
  - Cloudflare Clef (clef-flash) on:
      * Mac / Metal GPU (and CPU) via LiteRT (`ai-edge-litert` CompiledModel, `.tflite` rowprefill graphs)
      * Linux / NVIDIA CUDA, Apple Silicon MPS, and CPU via PyTorch (`joint_schema_model.py`)
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import threading
import torch

MAX_TOKENS = 8192
_GPU_LOCK = threading.Lock()  # module-level lock across all scorer instances on the same GPU
REPO_ROOT = pathlib.Path(__file__).resolve().parent


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_secjev_work() -> pathlib.Path:
    env_work = os.environ.get("SECJEV_WORK")
    if env_work:
        return pathlib.Path(env_work).expanduser().resolve()
    default_work = pathlib.Path.home() / "secjev-4b-work"
    if default_work.is_dir():
        return default_work.resolve()
    raise RuntimeError(
        "SECJEV_WORK environment variable is not set and ~/secjev-4b-work was not found. "
        "Set SECJEV_WORK=/path/to/secjev-4b-work when loading a Kev/SecJev checkpoint."
    )


def _has_mlx() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


def _is_clef_target(model_dir: str | pathlib.Path) -> bool:
    s = str(model_dir)
    if s.startswith("clef:") or "clef-flash" in s.lower() or s.endswith(".tflite"):
        return True
    p = pathlib.Path(s).expanduser()
    if p.is_dir():
        if (p / "contract.json").is_file() or (p / "host" / "clef_litert").is_dir():
            return True
        if (p / "joint_schema_model.py").is_file() and (
            (p / "joint_head.safetensors").is_file() or (p / "head" / "joint_head.safetensors").is_file()
        ):
            return True
        if any(p.glob("clef-flash_rowprefill_embeds_*.tflite")):
            return True
    return False


def _resolve_clef_litert_dir(model_dir: str | pathlib.Path) -> pathlib.Path | None:
    """Resolve a LiteRT Clef folder if `model_dir` is a `.tflite` file or a LiteRT directory."""
    s = str(model_dir).removeprefix("clef:")
    p = pathlib.Path(s).expanduser().resolve()
    if p.is_file() and p.suffix == ".tflite":
        # Check if parent already has LiteRT auxiliary files (head/, tables/, tokenizer/)
        if (p.parent / "head" / "joint_head.safetensors").is_file():
            return p.parent
        # Otherwise use checkpoints/clef-flash-LiteRT and ensure the .tflite is linked inside it
        fallback = REPO_ROOT / "checkpoints" / "clef-flash-LiteRT"
        if (fallback / "head" / "joint_head.safetensors").is_file():
            target_link = fallback / p.name
            if not target_link.exists():
                target_link.symlink_to(p)
            return fallback
        raise FileNotFoundError(
            f"LiteRT graph {p} requires head/, tables/, and tokenizer/ in {p.parent} or {fallback}"
        )
    if p.is_dir() and any(p.glob("clef-flash_rowprefill_embeds_*.tflite")) and (
        p / "head" / "joint_head.safetensors"
    ).is_file():
        return p
    return None


class Clef:
    """Cross-platform Cloudflare Clef scorer supporting LiteRT (Mac Metal / CPU) and PyTorch (CUDA / MPS / CPU)."""

    def __init__(
        self,
        model_dir: str | pathlib.Path = "checkpoints/clef-flash-LiteRT",
        device: str | None = None,
        backend: str | None = None,
    ):
        raw_target = str(model_dir).removeprefix("clef:")
        self.device = device or os.environ.get("CLEF_DEVICE") or default_device()
        litert_dir = _resolve_clef_litert_dir(raw_target)
        # Auto-select backend:
        # - If CLEF_BACKEND is set ("litert" or "torch"), honor it.
        # - Otherwise, use "litert" when a LiteRT folder/tflite is provided on macOS/Metal or CPU,
        #   and use "torch" on CUDA or when pointed at a PyTorch checkpoint/HF repo.
        env_backend = (backend or os.environ.get("CLEF_BACKEND") or "").lower()
        if env_backend in ("litert", "tflite"):
            use_litert = True
        elif env_backend in ("torch", "pytorch", "cuda"):
            use_litert = False
        else:
            use_litert = litert_dir is not None and self.device != "cuda"

        self._lock = _GPU_LOCK
        if use_litert:
            if litert_dir is None:
                raise FileNotFoundError(f"Could not find Clef LiteRT directory or .tflite for {model_dir!r}")
            host_pkg_parent = str(litert_dir / "host")
            if host_pkg_parent not in sys.path:
                sys.path.insert(0, host_pkg_parent)
            from clef_litert.errors import RequestTooLong, SchemaTooLong
            from clef_litert.response import question_probs
            from clef_litert.runtime import ClefLiteRT

            self._RequestTooLong = (RequestTooLong, SchemaTooLong)
            self._question_probs = question_probs
            accel = os.environ.get("CLEF_ACCELERATOR")
            if not accel:
                accel = "cpu" if self.device == "cpu" else "gpu"
            threads = int(os.environ.get("CLEF_THREADS", "8"))
            self.clef = ClefLiteRT.from_dir(litert_dir, accelerator=accel, threads=threads)
            self.max_tokens = max(self.clef.graphs.lengths)
            # Set host max_length to max_tokens + 1 so requests > max_tokens trigger RequestTooLong
            # instead of being silently truncated inside encode_record.
            self.clef.host.max_length = self.max_tokens + 1
            self.backend = f"litert-{accel}"
            # Optionally pre-compile the largest row graph at startup unless CLEF_LAZY_COMPILE=1
            if os.environ.get("CLEF_LAZY_COMPILE") != "1":
                self.clef.graphs._row_graph(self.max_tokens)
        else:
            # PyTorch backend (Linux/NVIDIA CUDA, Apple Silicon MPS, or CPU)
            p = pathlib.Path(raw_target).expanduser()
            jsm_path = p / "joint_schema_model.py" if p.is_dir() else REPO_ROOT / "checkpoints" / "clef-flash-LiteRT" / "joint_schema_model.py"
            if not jsm_path.is_file():
                from huggingface_hub import hf_hub_download
                jsm_path = pathlib.Path(hf_hub_download(repo_id="Cloudflare/clef-flash", filename="joint_schema_model.py"))
            spec = importlib.util.spec_from_file_location("joint_schema_model", jsm_path)
            jsm = importlib.util.module_from_spec(spec)
            sys.modules["joint_schema_model"] = jsm
            spec.loader.exec_module(jsm)
            self._jsm = jsm
            dtype_name = os.environ.get("CLEF_DTYPE", "bfloat16" if self.device in ("cuda", "mps") else "float32")
            dtype = getattr(torch, dtype_name)
            repo_or_dir = raw_target if (p.is_dir() and (p / "config.json").is_file()) else os.environ.get("CLEF_TORCH_MODEL", "Cloudflare/clef-flash")
            self.model, self.processor = jsm.load_release_model(repo_or_dir, device=self.device, dtype=dtype)
            self.max_tokens = int(os.environ.get("CLEF_MAX_TOKENS", str(MAX_TOKENS)))
            self.backend = f"torch-{self.device}"

    def score(self, request: dict) -> dict:
        """Return {question_id: {candidate: probability}}. Raises ValueError('... exceeds ...') if input exceeds window."""
        req = {"model": "clef-flash", **request} if "model" not in request else request
        if self.backend.startswith("litert"):
            try:
                enc = self.clef.host.encode(req)
            except self._RequestTooLong as e:
                raise ValueError(f"state + question exceeds {self.max_tokens} tokens ({e})") from e
            with self._lock:
                rows = self.clef.graphs.tower(enc.tile, enc.patches) if enc.patches is not None else None
                hidden = self.clef.graphs.row_graph(enc.length, self.clef.host.row_inputs(enc, rows))
                logits = self.clef.host.scores(enc, hidden)
            return {
                q.question_id: dict(zip(q.option_ids, self._question_probs(z).tolist()))
                for q, z in zip(enc.record.questions, logits)
            }
        else:
            jsm = self._jsm
            try:
                encoded = jsm.encode_record(
                    self.processor.tokenizer,
                    req,
                    max_length=self.max_tokens + 1,
                    processor=self.processor,
                )
            except ValueError as e:
                if "maximum is" in str(e):
                    raise ValueError(f"state + question exceeds {self.max_tokens} tokens ({e})") from e
                raise
            if len(encoded.input_ids) > self.max_tokens:
                raise ValueError(f"state + question exceeds {self.max_tokens} tokens")
            with self._lock, torch.inference_mode():
                batch = jsm.collate_records([encoded], self.processor.tokenizer.pad_token_id, torch.device(self.device))
                logits = self.model(batch)[0]
            return {
                q.question_id: dict(zip(q.option_ids, z.float().softmax(-1).cpu().tolist()))
                for q, z in zip(encoded.questions, logits)
            }


class SecJev:
    def __init__(self, model_dir: str, device: str | None = None):
        if _is_clef_target(model_dir):
            self._clef = Clef(model_dir, device=device)
            self.device = self._clef.device
            self.backend = self._clef.backend
            return
        self._clef = None
        work = _resolve_secjev_work()
        kev_src = str(work / "source/kev")
        if kev_src not in sys.path:
            sys.path.insert(0, kev_src)
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.model import rows_of

        self._SystemOneRequest = SystemOneRequest
        self._to_record = to_record
        self._rows_of = rows_of

        device = device or default_device()
        tf32 = os.environ.get("SECJEV_TF32") == "1"  # off = authors' reference numerics
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        ckpt = Checkpoint(model_dir)
        kev_backend = os.environ.get("KEV_BACKEND", "mlx" if (device == "mps" and _has_mlx()) else "torch")
        base_dir = work / "base"
        if base_dir.exists() and kev_backend != "mlx":
            ckpt.meta.base = str(base_dir)
            ckpt.meta.base_revision = None
        if device == "mps" and kev_backend == "mlx":
            opts = LoadOptions(backend="mlx")
        else:
            opts = LoadOptions(backend="torch", dtype=torch.float32, merge=False, attn="sdpa")
        self.tok, self.model = ckpt.load(device, opts)
        self.device = device
        self.backend = f"kev-{kev_backend}-{device}"
        self._lock = _GPU_LOCK  # shared lock prevents concurrent Triton autotuning / multi-model OOM

    def score(self, request: dict) -> dict:
        """Return {question_id: {candidate: probability}}. Raises if input exceeds the model's window."""
        if self._clef is not None:
            return self._clef.score(request)
        req = self._SystemOneRequest.model_validate(request)
        record, metadata = self._to_record(req)
        enc = self.model.encode(self.tok, record, max_state=MAX_TOKENS, max_branch=MAX_TOKENS, strict=True)
        state, _, branches = self._rows_of(enc)
        if any(len(state) + len(b["ids"]) > MAX_TOKENS for b in branches):
            raise ValueError("state + question exceeds 8192 tokens")
        with self._lock, torch.inference_mode():
            logits = self.model.forward(enc)
        probs = [torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
        return {m["id"]: dict(zip(m["keys"], p)) for m, p in zip(metadata, probs)}
