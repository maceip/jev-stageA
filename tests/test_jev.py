"""Compact maximal-coverage tests; only GPU/network boundaries are stubbed."""
import gzip, json, runpy, sys, tempfile, types, unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock
R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
NS = types.SimpleNamespace
_SAVED_MODULES = dict(sys.modules)  # restored after import; fakes are re-installed by setUpModule
def mod(name, **kw):
    m = types.ModuleType(name)
    for k, v in kw.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m
class Abort(Exception):
    def __init__(s, msg, msgs, env): super().__init__(msg); s.messages, s.task_environment = msgs, env
class El:
    def __init__(s, *a, **k): pass
    def query(s, q, rt, env, msgs, ex): return q, rt, env, msgs, ex
class Exec(El):
    def query(s, q, rt, env, msgs, ex):
        return q, rt, env, list(msgs) + [{"role": "tool", "content": "ran", "tool_call": c, "error": None} for c in (msgs[-1].get("tool_calls") or [])], ex
class Pipe:
    MSGS = []
    def __init__(s, *a): s.els = a
    def query(s, *a): return None, None, a[2] if len(a) > 2 else None, Pipe.MSGS, None
class FCall:
    def __init__(s, function, args, id=None, placeholder_args=None): s.function, s.args, s.id, s.placeholder_args = function, args, id, placeholder_args
class FRuntime:
    def __init__(s, tools=None): pass
    def run_function(s, env, fn, args): env.calls.append(fn)
class FEnv:
    def __init__(s): s.calls = []; s.inbox = NS(sent=[])
    def model_copy(s, deep=False): e = FEnv(); e.calls = list(s.calls); return e
class FTask:
    def __init__(s, kind="user", prompt="do task", goal="do evil", id="t1"): s.kind, s.PROMPT, s.GOAL, s.ID = kind, prompt, goal, id
    def init_environment(s, env): return env
class FSuite:
    name, data_path, tools = "workspace", ".", []
    def __init__(s): s.injection_tasks, s.user_tasks = {}, {"t1": FTask()}
    def get_user_task_by_id(s, i): return FTask("user")
    def get_injection_task_by_id(s, i): return FTask("inj")
    def load_and_inject_default_environment(s, inj): return FEnv()
    def get_injection_vector_defaults(s): return {"v1": "dflt"}
    def register_injection_task(s, cls): s.injection_tasks["injection_task_" + cls.__name__.replace("InjectionTask", "")] = cls
    def _check_task_result(s, task, out, pre, env, stack): return ("good_fn" in env.calls) if task.kind == "user" else ("bad_fn" in env.calls)
class FFixed:
    def __init__(s, template, suite, target_pipeline=None): s.jailbreak, s.task_suite = template, suite
    def get_injection_candidates(s, ut): return ["v1"]
class FakeEnc(dict):
    def to(s, d): return s
class FakeLogits:
    def __init__(s, n): s.n = n
    def float(s): return s
class FModel:
    config = NS(id2label={0: "benign", 1: "INJECTION"})
    def cuda(s): return s
    def to(s, d): return s
    def eval(s): return s
    def __call__(s, **kw): return NS(logits=FakeLogits(kw.get("_n", 1)))
def ftok(texts, **kw): return FakeEnc(_n=len(texts) if isinstance(texts, list) else 1)
ftok.from_pretrained = lambda *a, **k: ftok
FModel.from_pretrained = staticmethod(lambda *a, **k: FModel())
class Ctx:
    def __enter__(s): return s
    def __exit__(s, *a): return False
torch = mod("torch", softmax=lambda x, d: [[0.25, 0.75]] * x.n, inference_mode=lambda: Ctx(),
            device=lambda d: d, cuda=NS(is_available=lambda: False),
            backends=NS(mps=NS(is_available=lambda: False)))
mod("transformers", AutoModelForSequenceClassification=FModel, AutoTokenizer=NS(from_pretrained=lambda *a, **k: ftok))
class FChat:
    NEXT = "YES"
    class _C: pass
    def __init__(s, **kw):
        s.base_url = kw.get("base_url", "")
        s.chat = NS(completions=NS(create=lambda **k: NS(choices=[NS(message=NS(content=FChat.NEXT))])))
mod("openai", OpenAI=FChat)
class YErr(Exception): pass
mod("yaml", safe_load=lambda t: {"env": t}, YAMLError=YErr)
ad = mod("agentdojo"); ad.__path__ = []
mod("agentdojo.functions_runtime", FunctionCall=FCall, FunctionsRuntime=FRuntime)
mod("agentdojo.task_suite"); mod("agentdojo.task_suite.load_suites", get_suites=lambda v="v1.2.2": {"workspace": FSuite()})
mod("agentdojo.task_suite.task_suite", read_suite_file=lambda *a: "x: {v1}\n", functions_stack_trace_from_messages=lambda m: [])
mod("agentdojo.agent_pipeline", AbortAgentError=Abort, ToolsExecutor=Exec, AgentPipeline=Pipe, InitQuery=El, SystemMessage=El, ToolsExecutionLoop=El, GroundTruthPipeline=Pipe)
mod("agentdojo.agent_pipeline.base_pipeline_element", BasePipelineElement=El)
mod("agentdojo.agent_pipeline.agent_pipeline", load_system_message=lambda x: "sys")
mod("agentdojo.types", ChatToolResultMessage=lambda **kw: kw, text_content_block_from_string=lambda s: {"type": "text", "content": s})
mod("agentdojo.attacks"); mod("agentdojo.attacks.base_attacks", FixedJailbreakAttack=FFixed)
mod("agentdojo.base_tasks", BaseInjectionTask=object, BaseUserTask=object)
mod("agent_llm", OpenAILLM=lambda *a, **k: NS(kind="chat"))
mod("responses_llm", ResponsesLLM=lambda *a, **k: NS(kind="responses"))
class FakeSJ:
    def __init__(s, d=None): pass
    def score(s, req): return {q: ({"true": 0.7, "false": 0.3} if v.get("type") == "noul" else {"0": 0.1, "1": 0.2, "2": 0.7}) for q, v in req["questions"].items()}
mod("secjev_scorer", SecJev=FakeSJ)
import score, replay, analyze, defenses, attacks_ood, adaptive, report, make_paraphrases
_FAKE_MODULES = {k: v for k, v in sys.modules.items() if _SAVED_MODULES.get(k) is not v}
def _swap_modules(mapping, fallback):
    for k in _FAKE_MODULES:
        if k in mapping: sys.modules[k] = mapping[k]
        elif k in fallback: sys.modules[k] = fallback[k]
        else: sys.modules.pop(k, None)
_swap_modules({}, _SAVED_MODULES)  # keep collection of sibling test modules on the real modules
_RUNTIME_SAVED = {}
def setUpModule():
    _RUNTIME_SAVED.clear(); _RUNTIME_SAVED.update({k: sys.modules[k] for k in _FAKE_MODULES if k in sys.modules})
    sys.modules.update(_FAKE_MODULES)
def tearDownModule():
    _swap_modules(_RUNTIME_SAVED, _SAVED_MODULES)
CALL = {"function": "good_fn", "args": {"x": 1}, "id": "1"}
def msgs():
    return [{"role": "user", "content": "please help Emma"},
            {"role": "assistant", "content": None, "tool_calls": [CALL, {"function": "bad_fn", "args": {}, "id": "2"}]},
            {"role": "tool", "content": [{"type": "text", "content": "out1"}, {"type": "image", "content": "skip"}], "tool_call": CALL},
            {"role": "tool", "content": None, "error": "boom", "tool_call": {"function": "bad_fn", "args": {}, "id": "2"}},
            {"role": "assistant", "content": "done"}]
def ep(**kw):
    d = {"suite": "workspace", "user_task": "t1", "injection_task": "i1", "injections": {}, "utility": True,
         "attack_success": False, "error": None, "messages": msgs()}
    d.update(kw); return d


class ScoreTests(unittest.TestCase):
    def test_text_and_points(self):
        self.assertEqual(score.text_of(None), ""); self.assertEqual(score.text_of("a"), "a")
        self.assertEqual(score.text_of([{"type": "text", "content": "x"}, {"type": "y", "content": "z"}]), "x")
        pts = list(score.decision_points(msgs()))
        self.assertEqual([p[0] for p in pts], ["action", "action", "tool", "tool"])
        self.assertEqual(pts[2][3]["output"], "out1"); self.assertEqual(pts[3][3]["output"], "boom")
        self.assertEqual(len(pts[1][3]["history"]), 1); self.assertEqual(pts[2][3]["user"], "please help Emma")
    def test_requests(self):
        p = {"user": "u", "call": CALL, "output": "o", "history": [{"function": "f", "args": {}}],
             "seen": [{"function": "f", "output": "y" * 4000}] * 4}
        r = score.secjev_tool_request(p); self.assertEqual(r["state"]["untrusted_tool_response"], "o")
        a = score.secjev_action_request(p, True); self.assertEqual(len(a["state"]["untrusted_tool_outputs_seen"]), 4)
        self.assertEqual(len(a["state"]["untrusted_tool_outputs_seen"][0]["output"]), 4000)
        b = score.secjev_action_request(p, False); self.assertIn("provenance", b["state"]["untrusted_tool_outputs_seen"][0])
        self.assertEqual(len(b["state"]["untrusted_tool_outputs_seen"]), 4)
    def test_cache(self):
        k = score.cache_key("tool", {"a": 1}); self.assertEqual(k, score.cache_key("tool", {"a": 1}))
        c = {}; self.assertEqual(score.cached(c, k, lambda: 5), 5); self.assertEqual(score.cached(c, k, lambda: 9), 5)
        self.assertEqual(score.dedup_stats(c), {"unique": 1, "hits": 1})
    def test_hf_classifier(self):
        clf = score.HFClassifier("m", "injection")
        self.assertEqual(clf("hello"), 0.75); self.assertEqual(clf.predict_many(["a", "b"]), [0.75, 0.75])
        self.assertEqual(clf.predict_many([]), [])
    def test_judge(self):
        for ans, want in (("YES do it", 1.0), ("no way", 0.0)):
            FChat.NEXT = ans; self.assertEqual(score.LLMJudge("8000", "m")("u", "o"), want)
        FChat.NEXT = "maybe"
        with self.assertRaises(ValueError): score.LLMJudge("https://api.openai.com/v1", "m")("u", "o")
    def test_tool_score_chunking(self):
        sj = NS(score=lambda req: {"injection": {"true": 0.3}})
        v, n = score.secjev_tool_score(sj, {"user": "u", "call": CALL, "output": "x" * 40000})
        self.assertEqual((v, n > 1), (0.3, True))
        calls = {"n": 0}
        def flaky(req):
            calls["n"] += 1
            if calls["n"] < 3: raise ValueError("state exceeds limit")
            return {"injection": {"true": 0.9}}
        v, _ = score.secjev_tool_score(NS(score=flaky), {"user": "u", "call": CALL, "output": "abc"})
        self.assertEqual(v, 0.9)
        bad = NS(score=lambda req: (_ for _ in ()).throw(ValueError("state exceeds limit")))
        with self.assertRaises(ValueError): score.secjev_tool_score(bad, {"user": "u", "call": CALL, "output": "abc"})
        other = NS(score=lambda req: (_ for _ in ()).throw(ValueError("different failure")))
        with self.assertRaises(ValueError): score.secjev_tool_score(other, {"user": "u", "call": CALL, "output": ""})
    def test_action_score_dropping(self):
        p = {"user": "u", "call": CALL, "history": [{"function": "f", "args": {}}] * 2, "seen": [{"function": "f", "output": "o"}] * 2}
        calls = {"n": 0}
        def flaky(req):
            calls["n"] += 1
            if calls["n"] < 2: raise ValueError("state exceeds limit")
            return {"authorization": {"0": 0.1, "1": 0.2, "2": 0.7}}
        pr, dropped = score.secjev_action_score(NS(score=flaky), p, True)
        self.assertEqual((pr["2"], dropped), (0.7, 1))
        bad = NS(score=lambda req: (_ for _ in ()).throw(ValueError("state exceeds limit")))
        with self.assertRaises(ValueError): score.secjev_action_score(bad, dict(p, history=[], seen=[]), False)
        other = NS(score=lambda req: (_ for _ in ()).throw(ValueError("nope")))
        with self.assertRaises(ValueError): score.secjev_action_score(other, p, False)
    def test_score_episode_groups(self):
        e, pool = ep(), ThreadPoolExecutor(2)
        sj = NS(score=lambda req: {"injection": {"true": 0.2}, "authorization": {"0": 0.2, "1": 0.4, "2": 0.4}})
        out = score.score_episode(e, "secjev", {"sj": sj}, pool)
        self.assertAlmostEqual(out[0]["secjev_action_meta"], 0.6); self.assertEqual(out[2]["secjev_tool"], 0.2)
        out2 = score.score_episode(e, "classifiers", {"deberta": lambda t: 0.1, "piguard": lambda t: 0.9}, pool)
        self.assertEqual((out2[2]["deberta"], out2[2]["piguard"]), (0.1, 0.9))
        class FakeJudge:
            def __call__(s, u, o): return 1.0
            def judge_action(s, u, c, h, seen): return 1.0
        score._DP_CACHE.clear()
        out3 = score.score_episode(e, "judge", {"judge": FakeJudge()}, pool)
        self.assertEqual(out3[2]["llm_judge"], 1.0); self.assertEqual(out3[0]["llm_judge_action"], 1.0)
        class BoomJudge:
            def __call__(s, u, o): raise ValueError("unparseable judge answer")
            def judge_action(s, u, c, h, seen): raise ValueError("unparseable judge answer")
        score._DP_CACHE.clear()
        out4 = score.score_episode(e, "judge", {"judge": BoomJudge()}, pool)
        self.assertIn("llm_judge", out4[2]["errors"]); pool.shutdown()
    def test_score_main(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "benign.jsonl"
            run.write_text(json.dumps(ep(injection_task=None)) + "\n" + json.dumps(ep(error="x", user_task="t2")) + "\n")
            for group in ("secjev", "classifiers", "judge"):
                FChat.NEXT = "NO"
                with mock.patch.object(sys, "argv", ["s", str(run), "--group", group, "--outdir", d]):
                    score.main()
                rows = [json.loads(l) for l in open(Path(d) / group / "benign.0of1.jsonl")]
                self.assertEqual(len(rows), 1)
                with mock.patch.object(sys, "argv", ["s", str(run), "--group", group, "--outdir", d]):  # resume: no dupes
                    score.main()
                self.assertEqual(len(open(Path(d) / group / "benign.0of1.jsonl").readlines()), 1)


class ReplayAnalyzeTests(unittest.TestCase):
    def test_executed_calls(self):
        m = msgs()
        self.assertEqual(len(replay.executed_calls(m, None)), 2)
        self.assertEqual(len(replay.executed_calls(m, ("action", 1, 1))), 1)
        self.assertEqual(len(replay.executed_calls(m, ("action", 1, 0))), 0)
        self.assertEqual(len(replay.executed_calls(m, ("tool", 2, 0))), 2)
        self.assertEqual(len(replay.executed_calls(m, ("tool", 1, 0))), 0)
        self.assertEqual(replay.executed_calls([{"role": "user", "content": "x"}], None), [])
    def test_outcome_and_validate(self):
        u, s = replay.outcome(ep(), None); self.assertEqual((u, s), (True, True))
        u2, s2 = replay.outcome(ep(), ("action", 1, 1)); self.assertEqual((u2, s2), (True, False))
        self.assertEqual(replay.outcome(ep(injection_task=None), None), (True, None))
        m2 = msgs()[:-1]  # last message is a tool result
        self.assertEqual(replay.outcome(ep(messages=m2), None)[0], True)
        b = replay.redact_bounds(ep(), ("action", 1, 1))
        self.assertEqual(b["stop"], (True, False)); self.assertEqual(b["redact_asr"], (False, True))
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "r.jsonl"
            good = ep(utility=True, attack_success=True); bad = ep(utility=False, attack_success=False, user_task="t9")
            f.write_text("\n".join(json.dumps(x) for x in (good, bad, ep(error="e"))) + "\n")
            n, mism = replay.validate(str(f)); self.assertEqual((n, len(mism)), (2, 1))
            with mock.patch.object(sys, "argv", ["replay.py", str(f)]):
                runpy.run_path(str(R / "replay.py"), run_name="__main__")
    def test_wilson_and_flags(self):
        self.assertEqual(analyze.wilson(0, 0)[1] != analyze.wilson(0, 0)[1], True)  # nan
        p, lo, hi = analyze.wilson(5, 10); self.assertTrue(lo < p < hi)
        self.assertFalse(analyze.wilson_overlap((0.1, 0.0, 0.2), (0.9, 0.8, 1.0)))
        self.assertTrue(analyze.wilson_overlap((0.5, 0.4, 0.6), (0.55, 0.5, 0.7)))
        pts = [{"kind": "tool", "msg": 2, "call": 0, "secjev_tool": 0.8, "errors": {}},
               {"kind": "action", "msg": 3, "call": 0, "secjev_action_meta": 0.7, "errors": {}}]
        self.assertEqual(analyze.first_flag(pts, "secjev_tool", 0.5), ("tool", 2, 0))
        self.assertIsNone(analyze.first_flag(pts, "secjev_tool", 0.9))
        with self.assertRaises(RuntimeError):
            analyze.first_flag([{"kind": "tool", "msg": 1, "call": 0, "errors": {"llm_judge": "x"}}], "llm_judge", 0.5)
        fused = [{"kind": "tool", "msg": 1, "call": 0, "secjev_tool": 0.1}, {"kind": "action", "msg": 2, "call": 0, "secjev_action_meta": 0.6}]
        self.assertIsNone(analyze.first_flag_fused(fused, 0.5, 0.5))
        fused[0]["secjev_tool"] = 0.9; self.assertEqual(analyze.first_flag_fused(fused, 0.5, 0.5), ("action", 2, 0))
        fused[0]["secjev_tool"] = 0.0; fused[1]["secjev_action_content"] = 0.95
        self.assertEqual(analyze.first_flag_fused(fused, 0.5, 0.5), ("action", 2, 0))
    def _scores(self, d, stem, deberta=1.0, inj=None):
        pts = [{"kind": "tool", "msg": 1, "call": 0, "secjev_tool": 0.1, "deberta": deberta, "piguard": 0.1, "llm_judge": 0.0, "errors": {}},
               {"kind": "action", "msg": 2, "call": 0, "secjev_action_meta": 0.1, "secjev_action_content": 0.1, "llm_judge_action": 0.0, "errors": {}}]
        row = {"suite": "workspace", "user_task": "t1", "injection_task": inj, "variant": 0, "points": pts}
        for g in analyze.GROUPS:
            (Path(d) / g).mkdir(parents=True, exist_ok=True)
            (Path(d) / g / f"{stem}.0of1.jsonl").write_text(json.dumps(row) + "\n")
    def test_load_and_defended(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "benign.jsonl"; run.write_text(json.dumps(ep(injection_task=None)) + "\n")
            self._scores(d, "benign")
            eps, sc, err = analyze.load(run, Path(d)); self.assertEqual(err, 0)
            with mock.patch.object(analyze, "outcome", return_value=(False, False)):
                self.assertEqual(analyze.defended(eps, sc, "deberta", 0.5, {}, "benign"), (1, 0, 0))
                self.assertEqual(analyze.defended(eps, sc, "secjev_tool", 0.5, {}, "benign"), (1, 1, 0))
                self.assertIsNone(analyze.choose_u95(eps, sc, "deberta", {}, "benign"))
                self.assertEqual(analyze.choose_u95(eps, sc, "secjev_tool", {}, "benign", suites={"workspace"}), 0.11)
                self.assertEqual(analyze.defended_fused(eps, sc, 0.05, 0.05, {}, "benign"), (1, 0, 0))
                eps2 = dict(eps); k = next(iter(eps2)); eps2[k] = dict(eps2[k], error="x")
                self.assertEqual(analyze.defended_fused(eps2, sc, 0.05, 0.05, {}, "benign"), (0, 0, 0))
                self.assertEqual(analyze.defended(eps2, sc, "deberta", 0.5, {}, "benign"), (0, 0, 0))
            with self.assertRaises(RuntimeError): analyze.load_scores(Path(d), "missing")
            run2 = Path(d) / "x.jsonl"; run2.write_text(json.dumps(ep(injection_task=None, user_task="zz")) + "\n")
            self._scores(d, "x")
            with self.assertRaises(RuntimeError): analyze.load(run2, Path(d))
    def test_analyze_main(self):
        with tempfile.TemporaryDirectory() as d:
            runs, sc = Path(d) / "runs", Path(d) / "scores"; runs.mkdir(); sc.mkdir()
            (runs / "benign.jsonl").write_text(json.dumps(ep(injection_task=None)) + "\n")
            (runs / "important_instructions.jsonl").write_text(json.dumps(ep()) + "\n")
            self._scores(str(sc), "benign"); self._scores(str(sc), "important_instructions", inj="i1")
            with mock.patch.object(analyze, "outcome", return_value=(False, False)):
                for extra in ([], ["--cross-suite"]):
                    with mock.patch.object(sys, "argv", ["a", "--runs", str(runs), "--scores", str(sc),
                                                          "--out", str(Path(d) / "o.json")] + extra):
                        analyze.main()
            res = json.load(open(Path(d) / "o.json"))
            self.assertIn("important_instructions", res["attacks"]); self.assertIsNone(res["detectors"]["deberta"]["thresholds"]["u95"])


class DefensesAttacksTests(unittest.TestCase):
    def test_as_call_and_tool_gate(self):
        self.assertEqual(defenses.as_call(NS(model_dump=lambda mode=None: {"function": "f"}))["function"], "f")
        self.assertEqual(defenses.as_call(CALL), CALL)
        base = msgs()
        gate = defenses.ToolOutputGate(lambda u, c, o: 0.9, 0.5, "redact")
        trail = [base[0], {"role": "tool", "content": "bad", "tool_call": CALL}]
        _, _, _, out, _ = gate.query("q", None, FEnv(), [dict(m) for m in trail], {})
        self.assertIn("removed", out[-1]["content"][0]["content"]); self.assertEqual(out[-1]["flags"], ["redacted"])
        with self.assertRaises(Abort):
            defenses.ToolOutputGate(lambda u, c, o: 0.9, 0.5, "stop").query("q", None, FEnv(), trail, {})
        ok = defenses.ToolOutputGate(lambda u, c, o: 0.1, 0.5, "stop").query("q", None, FEnv(), trail + [{"role": "user", "content": "x"}], {})
        self.assertEqual(len(ok[3]), 3)
        with self.assertRaises(AssertionError): defenses.ToolOutputGate(lambda *a: 0, 0.5, "nope")
    def test_action_executor(self):
        sj = NS(score=lambda req: {"authorization": {"0": 0.8, "1": 0.1, "2": 0.1}})
        ex = defenses.ActionGuardedExecutor(sj, 0.5, True, "redact")
        self.assertEqual(ex.query("q", None, FEnv(), [], {})[3], [])
        self.assertEqual(len(ex.query("q", None, FEnv(), [{"role": "user", "content": "u"}], {})[3]), 1)
        self.assertEqual(len(ex.query("q", None, FEnv(), [{"role": "assistant", "content": "hi"}], {})[3]), 1)
        full = msgs()[:2]
        _, _, _, out, _ = ex.query("q", None, FEnv(), full, {})
        self.assertEqual(len(out), 4)  # two allowed calls executed via stub executor
        hist = [{"role": "user", "content": "u"}, {"role": "assistant", "tool_calls": [CALL]},
                {"role": "tool", "content": "o", "tool_call": CALL},
                {"role": "assistant", "tool_calls": [CALL]}]
        self.assertEqual(len(ex.query("q", None, FEnv(), hist, {})[3]), 5)
        bad_sj = NS(score=lambda req: {"authorization": {"0": 0.0, "1": 0.0, "2": 1.0}})
        _, _, _, out2, _ = defenses.ActionGuardedExecutor(bad_sj, 0.5, False, "redact").query("q", None, FEnv(), list(full[:1]) + [{"role": "assistant", "tool_calls": [CALL]}], {})
        self.assertEqual(out2[-1]["error"], defenses.BLOCKED)
        with self.assertRaises(Abort):
            defenses.ActionGuardedExecutor(bad_sj, 0.5, False, "stop").query("q", None, FEnv(), list(full[:1]) + [{"role": "assistant", "tool_calls": [CALL]}], {})
        with self.assertRaises(AssertionError): defenses.ActionGuardedExecutor(sj, 0.5, True, "nope")
    def test_templates_and_load(self):
        suite, ut, it = FSuite(), FTask("user"), FTask("inj")
        for name in ("bland_note", "metadata_field", "quoted_reply", "zero_width"):
            atk = attacks_ood.load(name, suite)
            out = atk.attack(ut, it); self.assertIn("v1", out)
        self.assertIn(attacks_ood.ZWSP, attacks_ood.load("zero_width", suite).attack(ut, it)["v1"])
        self.assertEqual(attacks_ood.load("bland_note", suite).summarize_task(ut), ut.PROMPT)
        ll = attacks_ood.load("llmail", FSuite())
        self.assertEqual(len(ll.payloads), 20)
        cls = attacks_ood._llmail_task_class(0); t = cls.__new__(cls)
        self.assertEqual(t.ground_truth(None)[0].function, "send_email")
        env = FEnv(); env.inbox.sent = [NS(recipients=["CONTACT@contact.com"], body="Confirmation!")]
        self.assertTrue(t.security(None, None, env)); self.assertFalse(t.security(None, None, FEnv()))
        payloads, ids = attacks_ood.register_llmail(FSuite()); self.assertEqual(len(ids), 20)
        out = ll.attack(ut, NS(PAYLOAD_INDEX=0, GOAL="g")); self.assertIn("v1", out)
    def test_yaml_helpers(self):
        self.assertTrue(attacks_ood._yaml_printable("A")); self.assertFalse(attacks_ood._yaml_printable("\x01"))
        self.assertTrue(attacks_ood._yaml_printable("\t")); self.assertTrue(attacks_ood._yaml_printable("é"))
        self.assertEqual(attacks_ood._dq('a"b\\c\nd\te\x01'), 'a\\"b\\\\c\\nd\\te\\u0001')
        self.assertEqual(attacks_ood._sq("a'b\nc"), "a''b\n\nc")
        self.assertEqual(sorted(attacks_ood._strings({"k": ["v", {"n": "w"}], "z": 3})), ["k", "n", "v", "w", "z"])
        suite = FSuite()
        out = attacks_ood.encode_injections(suite, {"v1": "PAYLOAD text"})
        self.assertEqual(out["v1"], "PAYLOAD text")
        with mock.patch.object(attacks_ood._yaml, "safe_load", side_effect=YErr("bad")):
            self.assertFalse(attacks_ood._contains("x: {v1}", {"v1": "d"}, "v1", "plain", "p"))
        class NoVec(FSuite):
            pass
        attacks_ood._encoders.cache_clear()
        with mock.patch.object(attacks_ood, "_read", return_value="novec"):
            with self.assertRaises(RuntimeError): attacks_ood._encoders(NoVec())
        attacks_ood._encoders.cache_clear()
        attacks_ood.encode_injections(suite, {"v1": "warm"})  # cache encoders for this suite
        with mock.patch.object(attacks_ood._yaml, "safe_load", return_value={"env": "empty"}):
            with self.assertRaises(RuntimeError): attacks_ood.encode_injections(suite, {"v1": "PAYLOAD text"})


class ReportParaphraseTests(unittest.TestCase):
    def _res(self):
        det = {d: {"thresholds": {"native": 0.5, "u95": None}, "benign_utility_native": (0.9, 0.8, 0.95)} for d in report.DET}
        row = {"n": 10, "undefended": {"asr": (0.2, 0.1, 0.3), "utility": (0.5, 0.4, 0.6)},
               "secjev_tool@native": {"asr": (0.0, 0.0, 0.1), "utility": (0.4, 0.3, 0.5)}}
        return {"benign": {"n": 10, "errors": 1, "utility_undefended": (0.9, 0.8, 0.95)},
                "detectors": det, "attacks": {"important_instructions": row, "custom_atk": row}}
    def test_report_tables(self):
        self.assertIn("90.0%", report.pct((0.9, 0.8, 0.95)))
        self.assertAlmostEqual(report.wilson(5, 10)[0], 0.5)
        md = report.agent_tables("agent", self._res(), "native")
        self.assertIn("Standard AgentDojo", md); self.assertIn("custom_atk", md); self.assertIn("n/a", md)
        md2 = report.agent_tables("agent", self._res(), "u95"); self.assertIn("n/a", md2)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "gpt_secjev_tool.jsonl"
            rows = [{"pair": ["w", "t", "i"], "error": None, "success": True, "rounds": 3},
                    {"pair": ["w", "t2", "i"], "error": None, "success": False, "rounds": 10},
                    {"pair": ["w", "t3", "i"], "error": "boom", "success": None, "rounds": 0}]
            f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            tab = report.adaptive_table([str(f)]); self.assertIn("secjev_tool", tab); self.assertIn("3.0", tab)
            rows[0]["success"] = False; f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.assertIn("–", report.adaptive_table([str(f)]))
            f.write_text(json.dumps({"pair": ["w", "t", "i"], "error": "e", "success": None}) + "\n")
            self.assertNotIn("gpt", report.adaptive_table([str(f)]))
            aj = Path(d) / "res.json"; aj.write_text(json.dumps(self._res()))
            rows[0]["success"] = True; (Path(d) / "a_b.jsonl").write_text(json.dumps(rows[0]) + "\n")
            with mock.patch.object(sys, "argv", ["report.py", str(aj), str(Path(d) / "a_b.jsonl")]):
                runpy.run_path(str(R / "report.py"), run_name="__main__")
    def test_paraphrase_required(self):
        req = make_paraphrases.required("Please email Emma Johnson at emma@x.com about $1,250 on 2024-05-01. See https://x.com/a and 'quoted bit'.")
        for tok in ("Emma", "emma@x.com", "1,250", "2024-05-01", "https://x.com/a"): self.assertIn(tok, req)
        self.assertNotIn("Please", req)
        self.assertIn("Emma", make_paraphrases.required("Emma Johnson wants lunch."))  # sentence-start proper noun kept
        orig = "Send 50 dollars to Alice."
        self.assertTrue(make_paraphrases.verify_paraphrase(orig, "Please send Alice the 50 dollars."))
        self.assertFalse(make_paraphrases.verify_paraphrase(orig, orig))
        self.assertFalse(make_paraphrases.verify_paraphrase(orig, "Send money somewhere."))
        self.assertFalse(make_paraphrases.verify_paraphrase(orig, 42))
    def test_paraphrase_main(self):
        suites = {s: NS(user_tasks={"t1": FTask(prompt="Send 50 dollars to Alice.", id="t1")}) for s in ("workspace", "travel", "banking", "slack")}
        valid = json.dumps(["Please send Alice the 50 dollars.", "Alice should get 50 dollars.", "Send 50 dollars to Alice today.", "Could you send Alice 50 dollars?"])
        seq = iter(["no json here", "[1,]"] + [valid] * 12)
        class PC:
            def __init__(s, **kw): s.chat = NS(completions=NS(create=lambda **k: NS(choices=[NS(message=NS(content=next(seq, valid)))])))
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "p.json")
            Path(out + ".progress.jsonl").write_text(json.dumps({"suite": "slack", "task": "t1", "paraphrases": ["Send Alice her 50 dollars."] * 4}) + "\n")
            with mock.patch.object(make_paraphrases.openai, "OpenAI", PC), mock.patch.object(make_paraphrases, "get_suites", return_value=suites), mock.patch.object(sys, "argv", ["m", "--model", "x", "--out", out]):
                make_paraphrases.main()
            self.assertEqual(sorted(json.load(open(out))), ["banking", "slack", "travel", "workspace"])
            class Bad:
                def __init__(s, **kw): s.chat = NS(completions=NS(create=lambda **k: NS(choices=[NS(message=NS(content="[]"))])))
            with mock.patch.object(make_paraphrases.openai, "OpenAI", Bad), mock.patch.object(make_paraphrases, "get_suites", return_value=suites), mock.patch.object(sys, "argv", ["m", "--model", "x", "--out", str(Path(d) / "q.json")]):
                with self.assertRaises(SystemExit): make_paraphrases.main()


class AdaptiveTests(unittest.TestCase):
    def test_gates_and_scorers(self):
        self.assertEqual(adaptive.as_call(NS(model_dump=lambda mode=None: {"function": "f"}))["function"], "f")
        self.assertEqual(adaptive.as_call(CALL), CALL)
        trail = [{"role": "user", "content": "u"}, {"role": "tool", "content": "bad", "tool_call": CALL}]
        self.assertEqual(len(adaptive.DetectorGate(lambda u, c, o: 0.1, 0.5, "stop").query("q", None, FEnv(), trail, {})[3]), 2)
        with self.assertRaises(Abort): adaptive.DetectorGate(lambda u, c, o: 0.9, 0.5, "stop").query("q", None, FEnv(), trail, {})
        sj = NS(score=lambda req: {"authorization": {"0": 0.9, "1": 0.05, "2": 0.05}})
        ag = adaptive.ActionGate(sj, 0.5, True, "stop")
        self.assertEqual(ag.query("q", None, FEnv(), [], {})[3], [])
        self.assertEqual(len(ag.query("q", None, FEnv(), [{"role": "user", "content": "u"}], {})[3]), 1)
        full = [{"role": "user", "content": "u"}, {"role": "assistant", "tool_calls": [CALL]},
                {"role": "tool", "content": "o", "tool_call": CALL}, {"role": "assistant", "tool_calls": [CALL]}]
        self.assertEqual(len(ag.query("q", None, FEnv(), full, {})[3]), 5)  # allowed call executed: +1 tool result
        bad = adaptive.ActionGate(NS(score=lambda req: {"authorization": {"0": 0.0, "1": 0.0, "2": 1.0}}), 0.5, False, "stop")
        with self.assertRaises(Abort): bad.query("q", None, FEnv(), full, {})
        self.assertIsNone(adaptive.make_scorer("none")); self.assertIsNone(adaptive.make_scorer("secjev_action_meta", sj=sj))
        self.assertEqual(adaptive.make_scorer("secjev_tool", sj=NS(score=lambda r: {"injection": {"true": 0.4}}))("u", CALL, "o"), 0.4)
        self.assertEqual(adaptive.make_scorer("llm_judge", judge=lambda u, o: 0.0)("u", CALL, "o"), 0.0)
        def boom(u, o): raise ValueError("x")
        self.assertEqual(adaptive.make_scorer("llm_judge", judge=boom)("u", CALL, "o"), 1.0)  # fails closed
        self.assertEqual(adaptive.make_scorer("deberta", clf=lambda o: 0.33)("u", CALL, "o"), 0.33)
        self.assertEqual(adaptive.wilson_width(0, 0), 1.0)
        self.assertTrue(adaptive.wilson_width(50, 100) < adaptive.wilson_width(5, 10))
        self.assertFalse(adaptive.should_stop_early(1, 5)); self.assertTrue(adaptive.should_stop_early(500, 1000, 0.10))
    def test_pipeline_candidates_oracle(self):
        p1 = adaptive.pipeline("https://api.openai.com/v1", "m", None, 0.5, "stop")
        self.assertIsInstance(p1, Pipe)
        p2 = adaptive.pipeline("http://localhost:8000/v1", "m", lambda u, c, o: 0.0, 0.5, "redact", sj=FakeSJ(), action="content")
        self.assertIsInstance(p2, Pipe)
        suite, ut = FSuite(), FTask("user")
        Pipe.MSGS = [{"role": "tool", "content": "see <CANARY---v1---> here", "tool_call": CALL}, {"role": "assistant", "content": None}]
        self.assertEqual(adaptive.candidates(suite, ut), ["v1"])
        self.assertEqual(adaptive.oracle_score(suite, ut, {}, None), 0.0)
        self.assertEqual(adaptive.oracle_score(suite, ut, {}, lambda u, c, o: 0.42), 0.42)
        Pipe.MSGS = [{"role": "assistant", "content": "x"}]
        self.assertEqual(adaptive.oracle_score(suite, ut, {}, lambda u, c, o: 0.42), 0.0)
        self.assertEqual(adaptive.candidates(suite, ut), [])
    def test_live(self):
        suite, ut, it = FSuite(), FTask("user"), FTask("inj")
        class GoodPipe:
            def query(s, q, rt, env): env.calls.append("good_fn"); return None, None, env, [{"role": "assistant", "content": "done", "tool_calls": [CALL]}], None
        self.assertEqual(adaptive.live(suite, GoodPipe(), ut, it, {}), (False, False, ["good_fn"]))
        class RedactPipe:
            def query(s, q, rt, env): return None, None, env, [{"role": "tool", "content": defenses.REDACTED, "tool_call": CALL}], None
        self.assertEqual(adaptive.live(suite, RedactPipe(), ut, it, {})[1], True)
        class AbortPipe:
            def query(s, q, rt, env): raise Abort("flag", [{"role": "assistant", "content": "x"}], env)
        self.assertEqual(adaptive.live(suite, AbortPipe(), ut, it, {})[1], True)
        class BadPipe:
            def query(s, q, rt, env): raise json.JSONDecodeError("bad", "doc", 0)
        h, a, calls = adaptive.live(suite, BadPipe(), ut, it, {})
        self.assertEqual((h, a, calls[0].startswith("agent_error")), (False, False, True))
    def _attacker(self, texts):
        it = iter(texts)
        return NS(chat=NS(completions=NS(create=lambda **k: NS(choices=[NS(message=NS(content=next(it)))]))))
    def test_attack_pair(self):
        args = NS(rounds=3, defense_desc="none", attacker_model="m")
        suite, ut, it = FSuite(), FTask("user"), FTask("inj")
        with mock.patch.object(adaptive, "candidates", return_value=["v1"]), mock.patch.object(adaptive, "live", return_value=(True, False, ["bad_fn"])):
            r = adaptive.attack_pair(args, suite, ut, it, None, None, 0.5, self._attacker(["junk", "<payload>evil</payload>"]))
        self.assertEqual((r["success"], r["rounds"], r["log"][0]["status"]), (True, 2, "attacker_no_payload"))
        with mock.patch.object(adaptive, "candidates", return_value=["v1"]), mock.patch.object(adaptive, "live", return_value=(False, True, [])):
            r2 = adaptive.attack_pair(NS(rounds=1, defense_desc="d", attacker_model="m"), suite, ut, it, None, None, 0.5, self._attacker(["<payload>x</payload>"]))
        self.assertEqual((r2["success"], r2["log"][0]["status"]), (False, "aborted"))
        with mock.patch.object(adaptive, "candidates", return_value=["v1"]), mock.patch.object(adaptive, "live", return_value=(False, False, ["good_fn"])):
            r3 = adaptive.attack_pair(NS(rounds=1, defense_desc="d", attacker_model="m"), suite, ut, it, None, None, 0.5, self._attacker(["<payload>x</payload>"]))
        self.assertEqual(r3["log"][0]["status"], "ignored")
    def test_adaptive_main(self):
        with tempfile.TemporaryDirectory() as d:
            pairs = Path(d) / "pairs.json"; pairs.write_text(json.dumps([["workspace", "t1", "i1"], ["bad", "t", "i"]]))
            FChat.NEXT = "<payload>evil</payload>"
            Pipe.MSGS = [{"role": "assistant", "content": "done"}]
            argv = ["a", "--pairs", str(pairs), "--model", "m", "--attacker-model", "m", "--rounds", "1", "--workers", "2", "--out", str(Path(d) / "o.jsonl")]
            with mock.patch.object(adaptive, "get_suites", return_value={"workspace": FSuite()}), mock.patch.object(adaptive, "candidates", return_value=["v1"]):
                with mock.patch.object(sys, "argv", argv + ["--defense", "none"]): adaptive.main()
                with mock.patch.object(sys, "argv", argv + ["--defense", "none"]): adaptive.main()  # done pairs skipped
            rows = [json.loads(l) for l in open(Path(d) / "o.jsonl")]
            ok = [r for r in rows if r["error"] is None]; bad = [r for r in rows if r["error"]]
            self.assertEqual((len(ok) >= 1, len(bad) >= 1), (True, True))
            empty = Path(d) / "e.json"; empty.write_text("[]")
            base = ["a", "--pairs", str(empty), "--model", "m", "--attacker-model", "m", "--out", str(Path(d) / "o2.jsonl")]
            with mock.patch.object(adaptive, "get_suites", return_value={"workspace": FSuite()}):
                for defense in ("secjev_tool", "secjev_action_content", "deberta", "piguard", "llm_judge"):
                    with mock.patch.object(sys, "argv", base + ["--defense", defense]): adaptive.main()
    def test_adaptive_main_sync_and_entrypoints(self):
        from concurrent.futures import Future
        class SyncEx:
            def __init__(s, n=None): pass
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def submit(s, fn, *a):
                f = Future()
                try: f.set_result(fn(*a))
                except Exception as e: f.set_exception(e)
                return f
        with tempfile.TemporaryDirectory() as d:
            pairs = Path(d) / "pairs.json"; pairs.write_text(json.dumps([["workspace", "t1", "i1"], ["workspace", "t1", "i1"]]))
            FChat.NEXT = "<payload>evil</payload>"; Pipe.MSGS = [{"role": "assistant", "content": "done"}]
            argv = ["a", "--pairs", str(pairs), "--model", "m", "--attacker-model", "m", "--rounds", "1",
                    "--defense", "none", "--early-stop-width", "0.5", "--out", str(Path(d) / "o.jsonl")]
            with mock.patch.object(adaptive, "get_suites", return_value={"workspace": FSuite()}), mock.patch.object(adaptive, "candidates", return_value=["v1"]), mock.patch.object(adaptive, "ThreadPoolExecutor", SyncEx), mock.patch.object(adaptive, "should_stop_early", return_value=True), mock.patch.object(sys, "argv", argv):
                adaptive.main()
            self.assertEqual(len(open(Path(d) / "o.jsonl").readlines()), 1)  # early stop cancelled the rest
            runf = Path(d) / "b.jsonl"; runf.write_text(json.dumps(ep(injection_task=None)) + "\n")
            with mock.patch.object(sys, "argv", ["score.py", str(runf), "--group", "judge", "--outdir", d]):
                FChat.NEXT = "NO"; runpy.run_path(str(R / "score.py"), run_name="__main__")
            with mock.patch.object(sys, "argv", ["adaptive.py", "--help"]):
                with self.assertRaises(SystemExit): runpy.run_path(str(R / "adaptive.py"), run_name="__main__")
            with mock.patch.object(sys, "argv", ["analyze.py", "--help"]):
                with self.assertRaises(SystemExit): runpy.run_path(str(R / "analyze.py"), run_name="__main__")
            with mock.patch.object(sys, "argv", ["make_paraphrases.py", "--help"]):
                with self.assertRaises(SystemExit): runpy.run_path(str(R / "make_paraphrases.py"), run_name="__main__")
    def test_verify_corpus(self):
        rows = [{"source": "agentdojo_tool_outputs", "state": "{}", "questions": json.dumps({"q": {"type": "noul", "label": True, "src": "agentdojo_tool_outputs"}})},
                {"source": "injecagent", "state": "{}", "questions": json.dumps({"q": {"type": "score", "label": "2", "src": "injecagent"}})},
                {"source": "other", "state": "{}", "questions": "{}"}]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "c.gz"
            with gzip.open(f, "wt") as g:
                for r in rows: g.write(json.dumps(r) + "\n")
            with mock.patch.object(sys, "argv", ["verify_corpus.py", "model", str(f)]):
                runpy.run_path(str(R / "verify_corpus.py"), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
