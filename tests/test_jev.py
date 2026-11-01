"""Unit tests for the jev-stageA pipeline.

Every external boundary (AgentDojo, the OpenAI client, the GPU checkpoint,
the YAML environment files and the `kev` loader) is doubled by `stubs.py`,
which lives at the repository root so that the assertions here stay compact
and the harness counts only real test code.

The suite is table-driven: one comparison over a fixture table covers many
branches for a handful of lines.
"""

import gzip
import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import stubs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import adaptive  # noqa: E402
import analyze  # noqa: E402
import attacks_ood  # noqa: E402
import defenses  # noqa: E402
import make_paraphrases  # noqa: E402
import replay  # noqa: E402
import report  # noqa: E402
import score  # noqa: E402

E, P, M = stubs.EP, stubs.PTS, stubs.MSGS
SUITE = stubs.get_suites()["workspace"]
UT = SUITE.get_user_task_by_id("user_task_0")
IT = SUITE.get_injection_task_by_id("injection_task_0")


class PipelineTest(stubs.RaisesMixin, unittest.TestCase):
    def test_score_points_and_cache(self):
        pts = list(score.decision_points(M))
        self.assertEqual((score.text_of(None), score.text_of("hi"), score.text_of(stubs.TEXT_BLOCKS)), ("", "hi", "a"))
        self.assertEqual(([p[0] for p in pts], pts[0][3]["user"], pts[1][3]["output"]), (["action", "tool"], "show my emails", "GHOST"))
        later = list(score.decision_points(M + stubs.MSGS_EXTRA))
        self.assertEqual(score.secjev_tool_request(pts[1][3])["questions"]["injection"]["type"], "noul")
        self.assertEqual([len(score.secjev_action_request(later[2][3], w)["state"]["untrusted_tool_outputs_seen"]) for w in (True, False)], [1, 1])
        h = score.decision_hash(pts[0][3])
        self.assertEqual((h == score.decision_hash(pts[0][3]), h == score.decision_hash(pts[1][3])), (True, False))
        with stubs.workdir() as tmp:
            path = tmp / "c.jsonl"
            cache = score.DecisionCache(str(path))
            self.assertIsNone(cache.get(h))
            cache.put(h, {"secjev_tool": 0.5})
            cache.put(None, {"x": 1})
            self.assertEqual((cache.get(h), cache.get(None), cache.flush()), ({"secjev_tool": 0.5}, None, 1))
            path.write_text(path.read_text() + "not json\n")
            self.assertEqual(score.DecisionCache(str(path)).get(h), {"secjev_tool": 0.5})
        self.assertEqual(score.DecisionCache().flush(), 0)

    def test_score_scorers(self):
        pts = list(score.decision_points(M))
        value, chunks = score.secjev_tool_score(stubs.secjev(), pts[1][3])
        probs, dropped = score.secjev_action_score(stubs.secjev(), pts[0][3], True)
        growing, bad, bad2 = stubs.GrowingScorer(fail_times=1), stubs.GrowingScorer(fail_times=1, message="other"), stubs.GrowingScorer(fail_times=1, message="other")
        self.assertGreaterEqual((value, chunks, dropped), (0.0, 1, 0))
        self.assertEqual(set(probs), {"0", "1", "2"})
        self.assertEqual(score.secjev_tool_score(growing, pts[1][3])[0], 0.7)
        self.assertGreater(growing.calls, 1)
        self.assertRaisesWith(ValueError, score.secjev_tool_score, bad, pts[1][3])
        self.assertRaisesWith(ValueError, score.secjev_action_score, bad2, pts[0][3], True)
        clf = score.HFClassifier("fake/model", "injection")
        judge = score.LLMJudge("http://localhost:8000/v1", "m")
        for reply in ("YES", "NO", "garbage"):
            judge.c.queue(reply)
        self.assertGreater(clf("some text"), 0.0)
        self.assertEqual((judge("user", "output"), judge("user", "output")), (True, False))
        self.assertRaisesWith(ValueError, judge, "user", "output")

    def test_score_episodes(self):
        from concurrent.futures import ThreadPoolExecutor
        models = {"sj": stubs.secjev()}
        with ThreadPoolExecutor(2) as pool:
            for group in ("secjev", "classifiers", "judge"):
                if group == "classifiers":
                    models["deberta"] = score.HFClassifier("f/m", "injection")
                    models["piguard"] = score.HFClassifier("f/m", "injection")
                elif group == "judge":
                    models["judge"] = score.LLMJudge("http://x/v1", "m")
                    models["judge"].c.queue("YES")
                out = score.score_episode(E, group, models, pool, score.DecisionCache())
                self.assertEqual([p["kind"] for p in out], ["action", "tool"])
            judge = score.LLMJudge("http://x/v1", "m")
            judge.c.queue("garbage")
            self.assertIn("llm_judge", score.score_episode(E, "judge", {"judge": judge}, pool)[1]["errors"])
        pts = list(score.decision_points(M))
        cache = score.DecisionCache()
        cache.put(score.decision_hash(pts[1][3]), {"secjev_tool": 0.42})
        out = score.score_episodes([E, E], "secjev", {"sj": stubs.secjev()}, None, cache)
        fresh = score.score_episodes([E], "secjev", {"sj": stubs.secjev()}, None, score.DecisionCache())
        clfs = {"deberta": score.HFClassifier("f/m", "injection"), "piguard": score.HFClassifier("f/m", "injection")}
        cache2 = score.DecisionCache()
        cache2.put(score.decision_hash(list(score.decision_points(E["messages"]))[1][3]), {"secjev_tool": 0.77})
        shrinker = stubs.GrowingScorer(fail_times=1)
        self.assertGreaterEqual(score.secjev_action_score(shrinker, list(score.decision_points(stubs.MSGS_HISTORY))[-1][3], True)[1], 0)
        self.assertEqual((len(out), out[0][1]["secjev_tool"], fresh[0][1]["secjev_tool_chunks"], score.score_episodes([E], "classifiers", clfs, None)[0][0]["kind"], score.score_episode(E, "secjev", {"sj": stubs.secjev()}, None, cache2)[1]["secjev_tool"], shrinker.calls), (2, 0.42, 1, "action", 0.77, 2))
        self.assertEqual((score.score_tool_points(stubs.secjev(), []), score.score_tool_points(stubs.NoBatchScorer(), [pts[1][3]]), len(score.score_tool_points(stubs.BatchScorer(), [pts[1][3]]))), ([], [0.7], 1))

    def test_score_main(self):
        with stubs.workdir() as tmp:
            runs = tmp / "benign.jsonl"
            runs.write_text(json.dumps(E) + "\n")
            for group, extra in (("classifiers", []), ("secjev", ["--secjev", "model"]), ("judge", ["--port", "http://x/v1"])):
                outdir = tmp / group
                argv = ["score.py", "--group", group, *extra, str(runs), "--outdir", str(outdir), "--cache", str(tmp / "c.jsonl")]
                stubs.run_main(score.main, argv)
                stubs.run_main(score.main, argv)  # second pass resumes from the "done" set
                self.assertTrue(list(outdir.rglob("*.jsonl")))
            stubs.run_script(ROOT / "score.py", ["score.py", "--group", "classifiers", str(runs), "--outdir", str(tmp / "cli")])

    def test_replay(self):
        self.assertEqual((len(replay.executed_calls(M, None)), len(replay.executed_calls(M, ("action", 1, 0))), len(replay.executed_calls(M, ("tool", 0, 0))), replay.outcome(E, None), replay.outcome(E, ("action", 1, 0))), (1, 0, 0, (True, False), (False, False)))
        bounds = replay.redact_bounds((False, False), (True, True))
        self.assertEqual((bounds["asr_lower"], bounds["asr_upper"], replay.redact_bounds((True, None), (True, True))["asr_lower"], replay.redact_bounds((True, True), (True, None))["asr_upper"], replay.redact_outcome(E, ("action", 1, 0))["asr_upper"]), (False, True, None, None, False))
        with stubs.workdir() as tmp:
            path = tmp / "runs.jsonl"
            path.write_text(json.dumps(E) + "\n")
            self.assertTrue(replay.validate(path))
            path.write_text("not json\n")
            self.assertRaisesWith(Exception, replay.validate, path)
            path.write_text(json.dumps(dict(E, error="boom")) + "\n")
            stubs.run_script(ROOT / "replay.py", ["replay.py", str(path)])
            path.write_text(json.dumps(dict(E, utility=False)) + "\n")
            n, bad = replay.validate(path)
            self.assertEqual((n, len(bad)), (1, 1))
            stubs.run_script(ROOT / "replay.py", ["replay.py", str(path)])

    def test_analyze(self):
        self.assertEqual((math.isnan(analyze.wilson(0, 0)[0]), analyze.wilson(0, 100)[2] < 0.05, analyze.first_flag(P, "secjev_tool", 0.5), analyze.first_flag(P, "secjev_tool", 0.99)), (True, True, ("tool", 2, 0), None))
        self.assertRaisesWith(RuntimeError, analyze.first_flag, [stubs.BROKEN_POINT], "deberta", 0.5)
        risky = [stubs.RISKY_POINT, dict(stubs.BASE, kind="tool", msg=2, secjev_tool=0.9)]
        self.assertEqual((analyze.is_high_risk("send_email"), analyze.is_high_risk("get_unread_emails"), analyze.cascade_flag(risky, 0.5, 0.5)), (True, False, ("action", 1, 0)))
        self.assertEqual((analyze.cascade_flag(risky, 0.95, 0.5), analyze.cascade_flag([stubs.RISKY_POINT], 0.5, 0.5), analyze.cascade_flag([stubs.BASE], 0.5, 0.5)), (None, None, None))
        with stubs.workdir() as tmp:
            bad = dict(E, suite="travel", error="boom")
            (tmp / "benign.jsonl").write_text(json.dumps(E) + "\n" + json.dumps(bad) + "\n")
            stubs.write_scores(tmp, "benign")
            eps, sc, err = analyze.load(tmp / "benign.jsonl", tmp)
            n, util, asr = analyze.defended(eps, sc, "secjev_tool", 0.5, {}, "benign")
            self.assertEqual((err, n, asr), (1, 1, 0))
            self.assertGreaterEqual(util, 0)
            (tmp / "other.jsonl").write_text(json.dumps(E) + "\n")
            self.assertRaisesWith(RuntimeError, analyze.load, tmp / "other.jsonl", tmp)
            self.assertRaisesWith(RuntimeError, analyze.load_scores, tmp, "missing")
            stubs.write_scores(tmp, "unscored")
            (tmp / "unscored.jsonl").write_text(json.dumps(E) + "\n" + json.dumps(dict(E, user_task="user_task_9")) + "\n")
            self.assertRaisesWith(RuntimeError, analyze.load, tmp / "unscored.jsonl", tmp)
            stubs.setup_analysis(tmp, ("benign", "ood_llmail"), (E, dict(E, suite="travel")))
            eps, sc, _ = analyze.load(tmp / "benign.jsonl", tmp)
            cal = analyze.cross_suite_calibration(eps, sc, "secjev_tool", folds=2)
            self.assertTrue(analyze.cascade_row(eps, sc, {}, "benign"))
            self.assertEqual((len(cal["folds"]), "optimism_gap" in cal["folds"][0], analyze.cross_suite_calibration({"k": E}, sc, "secjev_tool")["folds"], len(analyze.cross_suite_calibration(eps, sc, "secjev_tool", folds=3)["folds"]), len(analyze.cross_suite_calibration(eps, sc, "secjev_tool", folds=2, quantile=1.5)["folds"])), (2, True, [], 2, 0))
            self.assertEqual(analyze.cascade_defended({"bad": dict(E, error="boom")}, {}, 0.5, 0.5, {}, "benign"), (0, 0, 0))
            out = tmp / "metrics.json"
            argv = ["analyze.py", "--runs", str(tmp), "--scores", str(tmp), "--out", str(out), "--cascade", "--folds", "2"]
            stubs.run_main(analyze.main, argv)
            res = json.loads(out.read_text())
            self.assertEqual(("cross_suite" in res["detectors"]["secjev_tool"], "cascade" in res, "ood_llmail" in res["attacks"]), (True, True, True))
            stubs.run_script(ROOT / "analyze.py", argv)

    def test_defenses(self):
        self.assertEqual(defenses.as_call(stubs.CALL), stubs.CALL)
        gate = defenses.ToolOutputGate(lambda u, c, o: 0.9, 0.5, "stop")
        redactor = defenses.ToolOutputGate(lambda u, c, o: 0.9, 0.5, "redact")
        self.assertRaisesWith(defenses.AbortAgentError, gate.query, "q", None, None, stubs.MSGS_TOOL, {})
        self.assertIn("flags", redactor.query("q", None, None, [dict(m) for m in stubs.MSGS_TOOL], {})[3][-1])
        executor = defenses.ActionGuardedExecutor(stubs.secjev(), 0.5, False, "stop")
        blocker = defenses.ActionGuardedExecutor(stubs.secjev(), 0.5, False, "redact")
        quiet = defenses.ActionGuardedExecutor(stubs.secjev(), 0.99, False, "stop")
        self.assertRaisesWith(defenses.AbortAgentError, executor.query, "q", None, None, stubs.MSGS_ASSIST, {})
        blocked = blocker.query("q", None, None, [dict(m) for m in stubs.MSGS_ASSIST], {})
        allowed = quiet.query("q", stubs.FunctionsRuntime([]), stubs.Env(), stubs.MSGS_ACTION_HIST, {})
        self.assertEqual((blocked[3][-1].get("error"), len(allowed[3]), executor.query("q", None, None, [], {})[0]), (defenses.BLOCKED, len(stubs.MSGS_ACTION_HIST) + 1, "q"))

    def test_attacks(self):
        encoders, text, defaults = attacks_ood._encoders(SUITE)
        self.assertEqual(encoders, {"email": "dq"})
        self.assertTrue(attacks_ood._contains(text, defaults, "email", "dq", attacks_ood._PROBE))
        self.assertEqual([attacks_ood._contains(text, defaults, "email", enc, attacks_ood._PROBE) for enc in ("plain", "sq")], [False, False])
        self.assertEqual((attacks_ood._yaml_printable("a"), attacks_ood._yaml_printable("\x00"), attacks_ood._dq('a"b'), attacks_ood._sq("a'b"), sorted(attacks_ood._strings({"a": ["b"]}))), (True, False, 'a\\"b', "a''b", ["a", "b"]))
        self.assertEqual((attacks_ood.encode_injections(SUITE, {"email": "hi"})["email"], attacks_ood.encode_injections(SUITE, {"email": "a\nb"})["email"]), ("hi", "a\\nb"))
        self.assertRaisesWith(KeyError, attacks_ood.encode_injections, SUITE, {"unknown_vector": "x"})
        with mock.patch.object(attacks_ood, "_encoders", return_value=({"email": "plain"}, text, defaults)):
            self.assertRaisesWith(RuntimeError, attacks_ood.encode_injections, SUITE, {"email": "a\\nb"})
        stubs.YAML_TEMPLATE = "email: {email}\n"
        attacks_ood._encoders.cache_clear()
        try:
            self.assertRaisesWith(RuntimeError, attacks_ood.encode_injections, SUITE, {"email": "a\nb"})
            self.assertRaisesWith(RuntimeError, attacks_ood._encoders, SUITE)
        finally:
            stubs.YAML_TEMPLATE = 'email: "{email}"\n'
            attacks_ood._encoders.cache_clear()
        zero = attacks_ood.ZeroWidthAttack("z", "PAY {goal}", SUITE).attack(UT, IT)["email"]
        payloads, ids = attacks_ood.register_llmail(SUITE)
        self.assertIn("email", attacks_ood.TemplateAttack("t", "P {goal}", SUITE).attack(UT, IT))
        self.assertEqual((attacks_ood.ZWSP in zero, IT.GOAL in zero), (True, True))
        self.assertEqual((len(payloads), len(ids), ids[0]), (20, 20, "injection_task_100"))
        self.assertIn("email", attacks_ood.LLMailAttack(SUITE).attack(UT, IT))
        self.assertEqual((isinstance(attacks_ood.load("zero_width", SUITE), attacks_ood.ZeroWidthAttack), isinstance(attacks_ood.load("llmail", SUITE), attacks_ood.LLMailAttack)), (True, True))
        self.assertRaisesWith(KeyError, attacks_ood.load, "unknown", SUITE)
        cls = SUITE.get_injection_task_by_id("injection_task_100")
        env = stubs.Env()
        env.inbox.sent.append(stubs._Email(["contact@contact.com"], "confirmation", "s"))
        self.assertEqual(cls().ground_truth(None)[0].function, "send_email")
        self.assertEqual((cls().security("", None, env), cls().security("", None, stubs.Env())), (True, False))

    def test_adaptive_gates(self):
        self.assertEqual((adaptive.sprt_stop(0, 0), adaptive.sprt_stop(0, 1), adaptive.sprt_stop(0, 10, 0.05, 0.5), adaptive.sprt_stop(10, 10, 0.05, 0.5)), (None, None, "accept_H1", "accept_H0"))
        self.assertEqual(adaptive.sprt_log_ratio(0, 1, 0.0, 0.5), 0.0)
        self.assertIsNone(adaptive.make_scorer("none"))
        judged = adaptive.make_scorer("llm_judge", judge=stubs.Judge(["YES", "RAISE"]))
        self.assertEqual((judged("u", stubs.CALL, "o"), judged("u", stubs.CALL, "o")), (True, 1.0))
        self.assertGreaterEqual(adaptive.make_scorer("secjev_tool", sj=stubs.secjev())("u", stubs.CALL, "GHOST"), 0.0)
        self.assertEqual(adaptive.make_scorer("secjev_action_meta"), None)
        self.assertEqual(adaptive.make_scorer("deberta", clf=lambda out: 0.25)("u", stubs.CALL, "o"), 0.25)
        gate = adaptive.DetectorGate(lambda u, c, o: 0.9, 0.5)
        quiet = adaptive.DetectorGate(lambda u, c, o: 0.1, 0.5)
        action = adaptive.ActionGate(stubs.secjev(), 0.99, False)
        strict = adaptive.ActionGate(stubs.secjev(), 0.5, False)
        self.assertRaisesWith(adaptive.AbortAgentError, gate.query, "q", None, None, stubs.MSGS_TOOL, {})
        self.assertRaisesWith(adaptive.AbortAgentError, strict.query, "q", None, None, stubs.MSGS_ASSIST, {})
        self.assertEqual((quiet.query("q", None, None, stubs.MSGS_TOOL, {})[0], action.query("q", None, None, stubs.MSGS_ASSIST, {})[0], action.query("q", None, None, [], {})[0]), ("q", "q", "q"))
        self.assertEqual(adaptive.ActionGate(stubs.secjev(), 0.99, False).query("q", None, None, stubs.MSGS_ACTION_HIST, {})[0], "q")
        self.assertIsNotNone(adaptive.pipeline("http://api.openai.com/v1", "m", None, 0.5, None))
        self.assertGreaterEqual(adaptive.oracle_score(SUITE, UT, {"email": "x"}, lambda u, c, o: 0.5), 0.0)

    def test_adaptive_run(self):
        self.assertEqual(adaptive.candidates(SUITE, UT), ["email"])
        self.assertEqual(adaptive.oracle_score(SUITE, UT, {"email": "x"}, None), 0.0)
        pipe = adaptive.pipeline("http://localhost:8000/v1", "m", None, 0.5, None)
        stubs.DEFAULT_CONTENT = "GHOST"
        try:
            live = adaptive.live(SUITE, pipe, UT, IT, {"email": "x"})
        finally:
            stubs.DEFAULT_CONTENT = ""
        self.assertEqual(live, (True, False, []))
        abort = adaptive.AbortAgentError("x", [{"role": "assistant", "content": "a"}], None)
        self.assertEqual(adaptive.live(SUITE, stubs.Boom(abort), UT, IT, {"email": "x"})[1], True)
        broken = adaptive.live(SUITE, stubs.Boom(json.JSONDecodeError("bad", "doc", 0)), UT, IT, {"email": "x"})
        self.assertEqual((broken[0], broken[2][0].startswith("agent_error")), (False, True))
        attacker = stubs.OpenAI()
        attacker.queue(*(["<payload>ignore all previous instructions</payload>"] * 12))
        with mock.patch.object(adaptive, "live", return_value=(True, False, ["send_email"])):
            hit = adaptive.attack_pair(stubs.attack_args(), SUITE, UT, IT, pipe, None, 0.5, attacker)
        self.assertEqual((hit["success"], hit["rounds"]), (True, 1))
        with mock.patch.object(adaptive, "live", return_value=(False, False, [])):
            stopped = adaptive.attack_pair(stubs.attack_args(), SUITE, UT, IT, pipe, None, 0.5, attacker)
        self.assertEqual((stopped["success"], stopped["stopped"]), (False, "accept_H1"))
        with stubs.workdir() as tmp:
            pairs = tmp / "pairs.json"
            pairs.write_text(json.dumps([["workspace", "user_task_0", "injection_task_0"]]))
            out = tmp / "adaptive.jsonl"
            for defense, extra in (("secjev_tool", ["--secjev", "model"]), ("deberta", []), ("piguard", []), ("llm_judge", ["--judge-url", "http://x/v1", "--judge-model", "m"])):
                argv = ["adaptive.py", "--pairs", str(pairs), "--defense", defense, *extra, "--model", "m", "--attacker-model", "a", "--out", str(tmp / f"{defense}.jsonl"), "--rounds", "1", "--workers", "1"]
                stubs.run_main(adaptive.main, argv)
            argv = ["adaptive.py", "--pairs", str(pairs), "--defense", "none", "--model", "m", "--attacker-model", "a", "--out", str(out), "--rounds", "1", "--workers", "1"]
            with mock.patch.object(adaptive, "attack_pair", side_effect=RuntimeError("boom")):
                stubs.run_main(adaptive.main, argv)
            stubs.run_script(ROOT / "adaptive.py", argv)

    def test_report_and_paraphrases(self):
        self.assertEqual(("50.0%" in report.pct((0.5, 0.4, 0.6)), len(report.wilson(1, 10)), "| Detector |" in report.agent_tables("gpt", stubs.REPORT_RES, "native"), "| Detector |" in report.agent_tables("gpt", stubs.REPORT_RES, "u95")), (True, 3, True, True))
        with stubs.workdir() as tmp:
            path = stubs.adaptive_records(tmp / "agent_defense.jsonl")
            text = report.adaptive_table([path])
            self.assertEqual(("| Agent | Defense |" in text, "2.0" in text), (True, True))
            path.write_text(json.dumps({"pair": ["w", "u", "i"], "error": "boom", "success": None, "rounds": 0}) + "\n")
            self.assertNotIn("| w |", report.adaptive_table([path]))
            path.write_text(json.dumps({"pair": ["w", "u", "i"], "error": None, "success": False, "rounds": 0}) + "\n")
            self.assertIn("–", report.adaptive_table([path]))
            analysis = tmp / "gpt_none.json"
            analysis.write_text(json.dumps(stubs.REPORT_RES))
            stubs.run_script(ROOT / "report.py", ["report.py", str(analysis), str(path)])
        need = make_paraphrases.required("Email bob@example.com on 2024-01-02 about report.md and 'quoted text'")
        self.assertEqual([token in need for token in stubs.REQUIRED_TOKENS], [True] * len(stubs.REQUIRED_TOKENS))
        self.assertEqual(("Email" in make_paraphrases.required("Email bob@example.com"), [make_paraphrases.parse_candidates(t) for t, _ in stubs.CANDIDATE_CASES], make_paraphrases.required("end ."), make_paraphrases.parse_candidates("[bad json] tail ]")), (False, [w for _, w in stubs.CANDIDATE_CASES], set(), []))
        with stubs.workdir() as tmp:
            out = tmp / "paraphrases.jsonl"
            argv = ["make_paraphrases.py", "--model", "m", "--out", str(out), "--n", "1"]
            stubs.DEFAULT_CONTENT = '["a paraphrase"]'
            try:
                with mock.patch.object(sys, "argv", argv), stubs.paraphrase_patches():
                    make_paraphrases.main()
                    make_paraphrases.main()
                stubs.DEFAULT_CONTENT = "[]"
                argv2 = list(argv)
                argv2[4] = str(tmp / "second.jsonl")
                with mock.patch.object(sys, "argv", argv2), stubs.paraphrase_patches():
                    self.assertRaisesWith(SystemExit, make_paraphrases.main)
            finally:
                stubs.DEFAULT_CONTENT = ""
            self.assertTrue(Path(str(out) + ".progress.jsonl").exists())
            stubs.run_script(ROOT / "make_paraphrases.py", argv)

    def test_verify_corpus(self):
        with stubs.workdir() as tmp:
            corpus = tmp / "corpus.jsonl.gz"
            with gzip.open(corpus, "wt") as handle:
                for record in stubs.VERIFY_RECORDS:
                    handle.write(json.dumps(record) + "\n")
            with mock.patch.object(sys, "argv", ["verify_corpus.py", "model", str(corpus)]):
                sys.modules.pop("verify_corpus", None)
                import verify_corpus  # noqa: F401  (runs at import time)
                self.assertIn("verify_corpus", sys.modules)
