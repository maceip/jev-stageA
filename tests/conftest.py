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


# ---------------------------------------------------------------- fake/real module isolation
# test_jev.py imports stubs.py, whose install() replaces torch/agentdojo/kev/...
# in sys.modules with test doubles; the production modules imported afterwards
# stay bound to those doubles. The e2e tests in this directory need the real
# packages and real-bound production modules, so when both suites run in one
# pytest process the managed sys.modules entries are swapped per test file.
_MANAGED_TOP = {"torch", "transformers", "yaml", "openai", "tenacity", "agentdojo", "kev"}
_MANAGED_EXACT = {"score", "replay", "analyze", "defenses", "adaptive", "attacks_ood",
                   "report", "make_paraphrases", "secjev_scorer", "verify_corpus", "responses_llm"}
_module_cache = {}
_module_mode = None


def _managed(name):
    return name.split(".")[0] in _MANAGED_TOP or name in _MANAGED_EXACT


def _activate(mode):
    global _module_mode
    if mode == _module_mode:
        return
    if _module_mode is None and "stubs" not in sys.modules:
        _module_mode = mode  # doubles were never installed; nothing to swap
        return
    current = _module_mode or "fake"  # collection imported stubs first
    if current == mode:
        _module_mode = mode
        return
    _module_cache[current] = {k: v for k, v in sys.modules.items() if _managed(k)}
    for k in [k for k in sys.modules if _managed(k)]:
        del sys.modules[k]
    sys.modules.update(_module_cache.get(mode, {}))
    _module_mode = mode


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    name = Path(str(item.fspath)).name
    if name == "test_jev.py":
        _activate("fake")
    elif name.startswith("test_e2e"):
        _activate("real")


