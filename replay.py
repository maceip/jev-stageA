"""Stop-mode counterfactual outcomes by deterministic tool replay, conditional on recorded cuts.

Rebuild the environment, execute recorded calls before the cut, and use AgentDojo checkers with an
empty model output after abort. Exactness requires matching live decision inputs and deterministic tools;
validate() must reproduce recorded outcomes before using a recording for evidence. In particular, offline
batched action-content requests do not include earlier results from the same batch, unlike the live
sequential action monitor. This file does not establish parity for those decisions or predict redaction
continuations; redact_bounds reports only identification bounds.
"""
import json
from functools import lru_cache

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites


@lru_cache
def suites(version="v1.2.2"):
    import attacks_ood
    s = get_suites(version)
    attacks_ood.register_llmail(s["workspace"])
    return s


def executed_calls(messages, cut):
    """Calls with recorded results before a stop cut; never execute trailing unexecuted proposals.

    A tool-output gate runs after the whole ToolsExecutor batch, so every call in that batch has
    already executed even when the first result is flagged. Action gates cut within the batch.
    """
    calls = []
    ran = {(m.get("tool_call_id") or (m.get("tool_call") or {}).get("id")) for m in messages if m["role"] == "tool"}
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        results = []
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            results.append(messages[j])
            j += 1
        for k, c in enumerate(m.get("tool_calls") or []):
            if c.get("id") not in ran:  # proposed but never executed in the recording (e.g. run ended)
                continue
            if cut is not None:
                kind, ci, ck = cut
                if kind == "action" and (i, k) >= (ci, ck):
                    return calls
                if kind == "tool" and i >= ci:
                    return calls
            # Recorded results, not proposals, witness execution. Match ids when present; older
            # recordings may only carry the function/arguments. Consume matches exactly once.
            match = next((n for n, result in enumerate(results)
                          if (c.get("id") is not None and result.get("tool_call_id") == c["id"])
                          or (c.get("id") is None and result.get("tool_call", {}).get("function") == c["function"]
                              and result.get("tool_call", {}).get("args") == c["args"])), None)
            if match is not None:
                result = results.pop(match)
                if "blocked" not in result.get("flags", []):
                    calls.append(c)
    return calls


def redact_bounds(ep, cut):
    """Identification bounds, NOT confidence intervals or predicted redact benefits.

    With no intervention the observed continuation is unchanged. After any redaction/refusal,
    its subsequent actions and final answer are unobserved: both binary outcomes range over [0,1].
    Even a successful stop prefix is not generally irreversible, so no monotonicity is assumed.
    """
    if ep.get("error") is not None:
        raise ValueError("cannot bound outcomes for an errored episode")
    if ep.get("defense", "none") != "none":
        raise ValueError("redact counterfactual bounds require an undefended recording")
    identified = cut is None
    utility = (int(bool(ep["utility"])),) * 2 if identified else (0, 1)
    attack = None
    if ep.get("injection_task") is not None:
        attack = (int(bool(ep["attack_success"])),) * 2 if identified else (0, 1)
    return {"utility": utility, "attack_success": attack, "identified": identified}


def outcome(ep, cut, version="v1.2.2"):
    if ep.get("defense", "none") != "none":
        raise ValueError("stop replay requires an undefended recording, not a live defended trajectory")
    suite = suites(version)[ep["suite"]]
    ut = suite.get_user_task_by_id(ep["user_task"])
    it = suite.get_injection_task_by_id(ep["injection_task"]) if ep["injection_task"] else None
    env = ut.init_environment(suite.load_and_inject_default_environment(ep["injections"]))
    pre = env.model_copy(deep=True)
    rt = FunctionsRuntime(suite.tools)
    stack = []
    for c in executed_calls(ep["messages"], cut):
        rt.run_function(env, c["function"], c["args"])
        stack.append(FunctionCall(function=c["function"], args=c["args"], id=c.get("id")))
    if cut is None:
        last = ep["messages"][-1] if ep["messages"] else None
        out = last["content"] if last and last["role"] == "assistant" else []
    else:
        out = []
    out = out or []
    util = suite._check_task_result(ut, out, pre, env, stack)
    asr = suite._check_task_result(it, out, pre, env, stack) if it else None
    return bool(util), (bool(asr) if asr is not None else None)


def validate(path, version="v1.2.2"):
    bad, n = [], 0
    for line in open(path):
        ep = json.loads(line)
        if ep["error"] is not None:
            continue
        n += 1
        u, s = outcome(ep, None, version)
        if u != ep["utility"] or s != (ep["attack_success"] if ep["injection_task"] else None):
            bad.append((ep["suite"], ep["user_task"], ep["injection_task"], (u, s), (ep["utility"], ep["attack_success"])))
    return n, bad


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        n, bad = validate(p)
        print(p, "episodes", n, "mismatches", len(bad))
        for b in bad[:10]:
            print("  ", b)
