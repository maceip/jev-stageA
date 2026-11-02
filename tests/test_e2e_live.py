"""Real wiring and live tests through the OpenAI API (see conftest.py). Each SecJev-using step owns its GPU memory."""
import copy, json, os, re, subprocess, sys

import pytest

from conftest import ROOT, AGENT, ATTACKER, ATTACKER_IBAN, need, episodes, find, as_live

# ---------------------------------------------------------------- wiring
@pytest.mark.parametrize("spec", ["none", "cascade:content:0.5:stop"])
def test_build_defense_constructs_real_elements(spec):
    import run_agent, types
    need("SECJEV_WORK")
    ex, post = run_agent.build_defense(spec, types.SimpleNamespace(secjev=need("SECJEV_ACTION_CKPT", "~/ckpt/C1_e1")))
    from agentdojo.agent_pipeline import ToolsExecutor
    assert isinstance(ex, ToolsExecutor) and len(post) == (1 if spec.startswith("cascade") else 0)
    del ex, post
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def test_default_model_names_exist_on_the_api():
    import openai
    need("OPENAI_API_KEY")
    have = {m.id for m in openai.OpenAI().models.list().data}
    named = set()
    for f in ["run_agent.py", "adaptive.py", "score.py", "make_paraphrases.py"]:
        named |= set(re.findall(r'default="(gpt-[\w.\-]+)"', (ROOT / f).read_text()))
    assert named and named <= have, f"defaults not on the API: {sorted(named - have)}"


# ---------------------------------------------------------------- live, through the real OpenAI API
def _run(args, timeout=900):
    env = dict(os.environ, SECJEV_TF32="1")
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr[-2000:]


def test_live_run_agent_with_defense(tmp_path):
    need("OPENAI_API_KEY"); need("SECJEV_WORK")
    out = tmp_path / "live.jsonl"
    _run(["run_agent.py", "--attack", "ood:bland_note", "--suites", "banking", "--limit", "2", "--model", AGENT,
          "--defense", "action:content:0.35:redact", "--secjev", need("SECJEV_ACTION_CKPT", "~/ckpt/C1_e1"),
          "--workers", "2", "--out", str(out)])
    recs = [json.loads(l) for l in open(out)]
    assert len(recs) == 2 and all(r["error"] is None for r in recs), [r["error"] for r in recs]


@pytest.mark.parametrize("defense", ["none", "secjev_action_content", "secjev_cascade"])
def test_live_adaptive_round(tmp_path, defense):
    need("OPENAI_API_KEY")
    sj = []
    if defense != "none":
        need("SECJEV_WORK")
        sj = ["--secjev", need("SECJEV_ACTION_CKPT", "~/ckpt/C1_e1")]
    pairs, out = tmp_path / "p.json", tmp_path / "a.jsonl"
    pairs.write_text(json.dumps([["banking", "user_task_0", "injection_task_5"]]))
    _run(["adaptive.py", "--pairs", str(pairs), "--defense", defense, *sj,
          "--threshold", "0.35", "--mode", "redact", "--model", AGENT, "--attacker-model", ATTACKER,
          "--judge-model", AGENT, "--rounds", "1", "--workers", "1", "--out", str(out)])
    rec = json.loads(out.read_text().splitlines()[-1])
    assert rec["error"] is None and rec["log"], rec["error"]
