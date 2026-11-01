"""Record undefended AgentDojo trajectories against a local vLLM model.

Every (suite, user_task, injection_task) episode is run once with no defense; full message traces,
injections, utility and attack success are written as JSONL. Errors are recorded as errors and never
counted as secure or insecure.
"""
import argparse, json, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from pydantic import BaseModel

from agentdojo.agent_pipeline import (AbortAgentError, AgentPipeline, InitQuery, SystemMessage,
                                      ToolsExecutionLoop, ToolsExecutor)
from agent_llm import OpenAILLM  # upstream AgentDojo client: malformed tool calls raise, never dropped
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks import load_attack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages
from agentdojo.task_suite.load_suites import get_suites

import attacks_ood


DEFENSE_SPEC = "none"


def to_jsonable(o):
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    raise TypeError(type(o))


def build_defense(spec: str, a):
    """spec: 'none' | 'tool:<secjev_tool|piguard|deberta|llm_judge>:<thr>:<stop|redact>'
                    | 'action:<meta|content>:<thr>:<stop|redact>'  -> (executor, post_elements)"""
    if spec == "none":
        return ToolsExecutor(), []
    kind, name, thr, mode = spec.split(":")
    thr = float(thr)
    import defenses
    if kind == "action":
        from secjev_scorer import SecJev
        return defenses.ActionGuardedExecutor(SecJev(a.secjev), thr, name == "content", mode), []
    from score import HFClassifier, LLMJudge, secjev_tool_score
    if name == "secjev_tool":
        from secjev_scorer import SecJev
        sj = SecJev(a.secjev)
        scorer = lambda user, call, out: secjev_tool_score(sj, {"user": user, "call": call, "output": out})[0]
    elif name in ("piguard", "deberta"):
        clf = (HFClassifier("leolee99/PIGuard", "injection", remote=True) if name == "piguard"
               else HFClassifier("protectai/deberta-v3-base-prompt-injection-v2", "injection"))
        scorer = lambda user, call, out: clf(out)
    elif name == "llm_judge":
        judge = LLMJudge(a.judge_url, a.judge_model)
        scorer = lambda user, call, out: judge(user, out)
    else:
        raise ValueError(spec)
    return ToolsExecutor(), [defenses.ToolOutputGate(scorer, thr, mode)]


def build_pipeline(base_url: str, model: str, reasoning: str | None, executor, post) -> AgentPipeline:
    import os
    key = os.environ.get("OPENAI_API_KEY", "local")
    client = openai.OpenAI(base_url=base_url, api_key=key, timeout=900, max_retries=3)
    if "api.openai.com" in base_url:
        from responses_llm import ResponsesLLM
        llm = ResponsesLLM(client, model, reasoning or "low")
    else:
        llm = OpenAILLM(client, model, reasoning_effort=reasoning, temperature=0.0)
    p = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm,
                       ToolsExecutionLoop([executor, *post, llm])])
    p.name = "vllm_parsed"
    return p


def episode(suite, pipeline, user_task, injection_task, injections, variant=0, prompt=None):
    rec = {"suite": suite.name, "user_task": user_task.ID,
           "injection_task": injection_task.ID if injection_task else None, "injections": injections,
           "variant": variant}
    t0 = time.time()
    try:
        env = suite.load_and_inject_default_environment(injections)
        env = user_task.init_environment(env)
        pre = env.model_copy(deep=True)
        runtime = FunctionsRuntime(suite.tools)
        try:
            _, _, env, messages, _ = pipeline.query(prompt or user_task.PROMPT, runtime, env)
            rec["aborted"] = False
        except AbortAgentError as e:  # stop-mode live defense
            env, messages, rec["aborted"] = e.task_environment, e.messages, True
        out = (messages[-1]["content"] if messages[-1]["role"] == "assistant" else None) or []
        stack = functions_stack_trace_from_messages(messages)
        rec["utility"] = suite._check_task_result(user_task, out, pre, env, stack)
        rec["attack_success"] = (suite._check_task_result(injection_task, out, pre, env, stack)
                                 if injection_task else None)
        rec["messages"] = messages
        rec["error"] = None
        rec["defense"] = DEFENSE_SPEC
    except Exception as e:  # recorded, never scored
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, help="agentdojo attack name, ood:<name>, or 'benign'")
    ap.add_argument("--version", default="v1.2.2")
    ap.add_argument("--suites", default="workspace,travel,banking,slack,shopping,github,dailylife")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--defense", default="none")
    ap.add_argument("--secjev", default=None)
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--paraphrases", default=None, help="benign only: JSON from make_paraphrases.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="max episodes (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    global DEFENSE_SPEC
    DEFENSE_SPEC = a.defense
    executor, post = build_defense(a.defense, a)
    pipeline = build_pipeline(a.base_url, a.model, a.reasoning, executor, post)
    suites = get_suites(a.version)
    jobs = []
    for name in a.suites.split(","):
        suite = suites[name]
        if a.attack == "benign":
            para = json.load(open(a.paraphrases)) if a.paraphrases else {}
            for ut in suite.user_tasks.values():
                jobs.append((suite, ut, None, {}, 0, None))
                for k, p in enumerate(para.get(suite.name, {}).get(ut.ID, []), 1):
                    jobs.append((suite, ut, None, {}, k, p))
            continue
        attack = (attacks_ood.load(a.attack[4:], suite) if a.attack.startswith("ood:")
                  else load_attack(a.attack, suite, pipeline))
        if a.attack == "ood:llmail":
            targets = [suite.get_injection_task_by_id(t) for t in attack.task_ids]
        else:
            targets = [t for t in suite.injection_tasks.values()
                       if int(t.ID.rsplit("_", 1)[1]) < attacks_ood.LLMAIL_BASE_ID]
        for ut in suite.user_tasks.values():
            for it in targets:
                jobs.append((suite, ut, it, attack.attack(ut, it), 0, None))
    if a.limit:
        jobs = jobs[: a.limit]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)  # errored episodes are final too: no hidden retries
            done.add((r["suite"], r["user_task"], r["injection_task"], r.get("variant", 0)))
    jobs = [j for j in jobs if (j[0].name, j[1].ID, j[2].ID if j[2] else None, j[4]) not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already done)", flush=True)

    lock = threading.Lock()
    n_ok = n_err = 0
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        futs = [ex.submit(episode, j[0], pipeline, j[1], j[2], j[3], j[4], j[5]) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            with lock:
                f.write(json.dumps(r, default=to_jsonable) + "\n"); f.flush()
                n_ok += r["error"] is None; n_err += r["error"] is not None
            if i % 25 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} ok={n_ok} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
