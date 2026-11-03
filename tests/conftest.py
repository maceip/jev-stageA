"""Shared fixtures for the real end-to-end tests: real AgentDojo v1.2.2, real SecJev on the GPU, real OpenAI API.
No stubs, no mocks, no skips: a missing requirement fails the run.

Required environment:
  SECJEV_WORK          SecJev work dir (Qwen3.5-4B-Base + Kev source), as on the run boxes
  SECJEV_MODEL         released SecJev-4B checkpoint dir            (default <repo>/checkpoints/SecJev-4B)
  SECJEV_ACTION_CKPT   fine-tuned content action monitor (C1 ep1)   (default <repo>/checkpoints/SecJev-4B-Action-Monitor-C1)
  OPENAI_API_KEY       key with gpt-5.4-mini and gpt-5.5
Run from the repo root:  python -m pytest -q tests/
Models are module-scoped and released after each module, so the suite fits one 40 GB GPU.
"""
import copy, gc, json, os, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AGENT, ATTACKER = "gpt-5.4-mini", "gpt-5.5"
ATTACKER_IBAN = "US133000000121212121212"


def need(var, default=None):
    v = os.environ.get(var) or (default and os.path.expanduser(default))
    if not v or (var != "OPENAI_API_KEY" and not Path(v).exists()):
        pytest.fail(f"{var} is required for the real end-to-end tests (got {v!r})")
    return v


def _release():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


@pytest.fixture(scope="module")
def secjev():
    need("SECJEV_WORK", "~/secjev-4b-work")
    from secjev_scorer import SecJev
    m = SecJev(need("SECJEV_MODEL", str(ROOT / "checkpoints/SecJev-4B")))
    yield m
    del m
    _release()


@pytest.fixture(scope="module")
def action_monitor():
    need("SECJEV_WORK", "~/secjev-4b-work")
    from secjev_scorer import SecJev
    m = SecJev(need("SECJEV_ACTION_CKPT", str(ROOT / "checkpoints/SecJev-4B-Action-Monitor-C1")))
    yield m
    del m
    _release()


def episodes(rel):
    return [json.loads(l) for l in open(ROOT / rel)]


def as_live(msgs):
    """Recorded messages store tool calls as dicts; the live pipeline carries agentdojo FunctionCall objects."""
    from agentdojo.functions_runtime import FunctionCall
    return [dict(m, tool_calls=[FunctionCall(**c) for c in m["tool_calls"]]) if m.get("tool_calls") else m for m in msgs]


def find(eps, user_task, injection_task):
    return next(e for e in eps if e["user_task"] == user_task and e["injection_task"] == injection_task)


