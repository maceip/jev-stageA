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

from score import text_of


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


def _args_blob(call):
    return json.dumps(call.get("args") or {}, sort_keys=True, default=str)


def quotes_taint(text, blob, window=24):
    """True when a later call's arguments quote a distinctive window of a redacted tool output."""
    compact = " ".join((text or "").split())
    if len(compact) < 12:
        return bool(compact) and compact in blob
    step = max(8, window // 2)
    for start in range(0, max(1, len(compact) - window + 1), step):
        chunk = compact[start:start + window]
        if len(chunk) >= 12 and chunk in blob:
            return True
    return False


def redact_plan(messages, flags):
    """Calls that still run under redact mode, plus a bound tag.

    No flag: the executed set is exactly the recorded trajectory (`exact`).
    An action flag drops that call. A tool flag keeps the call that produced the output and drops later
    calls whose arguments quote a 24-character window of the redacted text (`dependent-calls-dropped`).
    This is a counterfactual bound, not a second live agent: the continuation the model would have
    written after seeing the notice is not invented here.
    """
    flagged = set(flags or [])
    tainted, calls, bound = [], [], "exact"
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            if message["role"] == "tool" and ("tool", i, 0) in flagged:
                bound = "dependent-calls-dropped"
                tainted.append(text_of(message.get("content")) or (message.get("error") or ""))
            continue
        for k, call in enumerate(message.get("tool_calls") or []):
            if ("action", i, k) in flagged:
                bound = "dependent-calls-dropped"
                continue
            if any(quotes_taint(text, _args_blob(call)) for text in tainted):
                bound = "dependent-calls-dropped"
                continue
            calls.append(call)
    return calls, bound


def outcome(ep, cut, version="v1.2.2", calls_override=None, empty_output=False):
    suite = suites(version)[ep["suite"]]
    ut = suite.get_user_task_by_id(ep["user_task"])
    it = suite.get_injection_task_by_id(ep["injection_task"]) if ep["injection_task"] else None
    env = ut.init_environment(suite.load_and_inject_default_environment(ep["injections"]))
    pre = env.model_copy(deep=True)
    rt = FunctionsRuntime(suite.tools)
    stack = []
    calls = calls_override if calls_override is not None else executed_calls(ep["messages"], cut)
    for c in calls:
        rt.run_function(env, c["function"], c["args"])
        stack.append(FunctionCall(function=c["function"], args=c["args"], id=c.get("id")))
    if calls_override is not None or empty_output:
        out = [] if empty_output or cut is not None else _recorded_output(ep)
    elif cut is None:
        out = _recorded_output(ep)
    else:
        out = []
    out = out or []
    util = suite._check_task_result(ut, out, pre, env, stack)
    asr = suite._check_task_result(it, out, pre, env, stack) if it else None
    return bool(util), (bool(asr) if asr is not None else None)


def _recorded_output(ep):
    if not ep["messages"]:
        return []
    last = ep["messages"][-1]
    return last["content"] if last["role"] == "assistant" else []


def redact_outcome(ep, flags, version="v1.2.2"):
    """Counterfactual utility and attack success for redact mode.

    Returns (utility, attack_success, info). info['bound'] is `exact` when nothing was redacted
    (identical to the recorded episode) and `dependent-calls-dropped` otherwise. Utility on the
    inexact bound uses an empty model output, matching stop-mode replay, after replaying only the
    calls that do not quote redacted text.
    """
    calls, bound = redact_plan(ep["messages"], flags)
    if bound == "exact":
        util, asr = outcome(ep, None, version)
    else:
        util, asr = outcome(ep, ("redact", 0, 0), version, calls_override=calls, empty_output=True)
    return util, asr, {"bound": bound, "executed": len(calls)}


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
