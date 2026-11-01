"""Exact counterfactual outcomes for abort-on-flag defenses, by deterministic tool replay.

A defense that aborts on its first flag leaves the trajectory identical up to that flag. We rebuild the
environment from the recorded injections, re-execute exactly the tool calls that would have run before the
abort, and score utility and attack success with AgentDojo's own checkers (model output empty: aborted).
validate() replays complete trajectories and must reproduce the recorded outcomes exactly.
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
    """Tool calls executed before an abort at `cut` = (kind, msg_index, call_index); cut=None means no abort."""
    calls = []
    ran = {(m.get("tool_call_id") or (m.get("tool_call") or {}).get("id")) for m in messages if m["role"] == "tool"}
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        for k, c in enumerate(m.get("tool_calls") or []):
            if c.get("id") not in ran:  # proposed but never executed in the recording (e.g. run ended)
                continue
            if cut is not None:
                kind, ci, ck = cut
                if kind == "action" and (i, k) >= (ci, ck):
                    return calls
                if kind == "tool" and i >= ci:  # tool result at ci was produced by an earlier assistant message
                    return calls
            calls.append(c)
    return calls


def outcome(ep, cut, version="v1.2.2"):
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
        last = ep["messages"][-1]
        out = last["content"] if last["role"] == "assistant" else []
    else:
        out = []
    out = out or []
    util = suite._check_task_result(ut, out, pre, env, stack)
    asr = suite._check_task_result(it, out, pre, env, stack) if it else None
    return bool(util), (bool(asr) if asr is not None else None)


def redact_bounds(pre_flag, undefended):
    """Bracket the outcome of a REDACTING defense from its two computable endpoints.

    `stop` mode is exactly computable by replay: the trajectory up to the
    flag is identical to the recording, so the outcome of the truncated
    trajectory IS the defended outcome. `redact` mode is not exactly
    computable, because the agent continues on a path the recording does
    not contain. What IS computable is a bracket:

      lower bound (certain): the outcome of the trajectory truncated at the
        flag. Everything before the flag executed identically. When the
        flag fires on a tool output, the injected text was removed before
        the agent read it, so the injected goal is unreachable from that
        point on: the lower bound for the attack success rate is 0.
      upper bound: the undefended outcome. Redaction only removes
        information; it cannot hand the agent a capability the undefended
        run did not have, so the attack cannot succeed at a higher rate
        than the undefended run.

    Returns a dict with the bracket. `pre_flag` and `undefended` are
    (utility, attack_success) pairs.
    """
    pre_u, pre_a = pre_flag
    full_u, full_a = undefended
    asr_upper = None if full_a is None else bool(full_a)
    asr_lower = None if (pre_a is None or full_a is None) else bool(pre_a) and bool(full_a)
    return {"utility_pre_flag": bool(pre_u),
            "utility_undefended": bool(full_u),
            "asr_lower": asr_lower,
            "asr_upper": asr_upper}


def redact_outcome(ep, cut, version="v1.2.2"):
    """Counterfactual bounds for a redacting defense on one episode."""
    return redact_bounds(outcome(ep, cut, version), outcome(ep, None, version))


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
