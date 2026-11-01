"""Live defenses for AgentDojo/AgentDyn pipelines, in two modes:

  stop   : abort the episode on the first flag (outcomes for this mode also come from exact replay)
  redact : tool-output detectors replace a flagged output with a fixed notice and the agent continues;
           the action monitor refuses only the flagged call (the agent sees a tool error) and continues.
A detector that cannot produce a verdict raises; nothing is silently passed through.
"""
from agentdojo.agent_pipeline import AbortAgentError, ToolsExecutor
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.types import ChatToolResultMessage, text_content_block_from_string

from score import text_of, secjev_tool_score, secjev_action_score

REDACTED = "[Content removed by a security filter: this tool output was flagged as containing injected instructions.]"
BLOCKED = "Error: this tool call was blocked by a security policy because it is not authorized by the user's request."


def as_call(c):
    return c.model_dump(mode="json") if hasattr(c, "model_dump") else c


def _user(messages):
    return next(text_of(m["content"]) for m in messages if m["role"] == "user")


class ToolOutputGate(BasePipelineElement):
    """Runs after ToolsExecutor: scores each new tool result."""

    def __init__(self, scorer, thr, mode):
        assert mode in ("stop", "redact")
        self.scorer, self.thr, self.mode = scorer, thr, mode

    def query(self, query, runtime, env, messages, extra_args):
        user = _user(messages)
        i = len(messages) - 1
        while i >= 0 and messages[i]["role"] == "tool":
            m = messages[i]
            out = m.get("error") or text_of(m["content"]) or ""
            if self.scorer(user, as_call(m["tool_call"]), out) >= self.thr:
                if self.mode == "stop":
                    raise AbortAgentError("defense flagged tool output", list(messages), env)
                m["content"] = [text_content_block_from_string(REDACTED)]
                m["error"] = None
                m.setdefault("flags", []).append("redacted")
            i -= 1
        return query, runtime, env, messages, extra_args


class ActionGuardedExecutor(ToolsExecutor):
    """Replaces ToolsExecutor. Scores every proposed call with the SecJev action monitor before it runs.
    stop: abort on the first flagged call. redact: the flagged call is not executed and the agent receives a tool
    error in its place; other calls run normally, in order. The assistant message is never altered."""

    def __init__(self, sj, thr, with_content, mode):
        super().__init__()
        assert mode in ("stop", "redact")
        self.sj, self.thr, self.with_content, self.mode = sj, thr, with_content, mode

    def query(self, query, runtime, env, messages, extra_args):
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1].get("tool_calls"):
            return query, runtime, env, messages, extra_args
        user = _user(messages)
        history, seen = [], []
        for m in messages[:-1]:
            if m["role"] == "assistant":
                history += [{"function": as_call(c)["function"], "args": as_call(c)["args"]} for c in (m.get("tool_calls") or [])]
            elif m["role"] == "tool":
                seen.append({"function": as_call(m["tool_call"])["function"],
                             "output": text_of(m["content"]) or (m.get("error") or "")})
        last = messages[-1]
        results = []
        for c in last["tool_calls"]:
            cd = as_call(c)
            pr, _ = secjev_action_score(self.sj, {"user": user, "call": cd, "history": list(history), "seen": list(seen)},
                                        self.with_content)
            flagged = pr["2"] + 0.5 * pr["1"] >= self.thr
            if flagged and self.mode == "stop":
                raise AbortAgentError("action monitor flagged a proposed tool call", [*messages, *results], env)
            if flagged:
                results.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string("")],
                                                     tool_call_id=cd.get("id"), tool_call=c, error=BLOCKED))
            else:
                single = dict(last, tool_calls=[c])
                _, _, env, out, _ = super().query(query, runtime, env, [*messages[:-1], single], extra_args)
                new = out[len(messages):]
                results += new
                seen += [{"function": cd["function"], "output": text_of(r["content"]) or (r.get("error") or "")} for r in new]
            history.append({"function": cd["function"], "args": cd["args"]})
        return query, runtime, env, [*messages, *results], extra_args


def build_secjev_cascade(sj, thr, with_content, mode):
    """Run the same 4B SecJev model at BOTH spots in the live loop:
      Spot 1 (ToolOutputGate): filters incoming tool outputs before the agent reads them.
      Spot 2 (ActionGuardedExecutor): checks outgoing tool calls before they execute."""
    scorer = lambda user, call, out: secjev_tool_score(sj, {"user": user, "call": call, "output": out})[0]
    return ActionGuardedExecutor(sj, thr, with_content, mode), [ToolOutputGate(scorer, thr, mode)]

