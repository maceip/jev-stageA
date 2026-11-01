"""AgentDojo LLM element backed by the OpenAI Responses API (needed for tool use on gpt-6.x models).

Converts AgentDojo's chat history to Responses input items and back. Malformed tool-call arguments raise
(json.JSONDecodeError) and are recorded as agent errors; nothing is dropped.
"""
import json
from collections.abc import Sequence

import openai
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string


def _text(m):
    return get_text_content_as_str(m["content"]) if m.get("content") else ""


def _call(c):
    return c if isinstance(c, FunctionCall) else FunctionCall(**c)


class ResponsesLLM(BasePipelineElement):
    def __init__(self, client: openai.OpenAI, model: str, reasoning_effort: str = "low"):
        self.client, self.model, self.effort = client, model, reasoning_effort
        self.name = model

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),
              messages: Sequence[ChatMessage] = [], extra_args: dict = {}):
        instructions, items = None, []
        for m in messages:
            r = m["role"]
            if r == "system":
                instructions = _text(m)
            elif r == "user":
                items.append({"role": "user", "content": _text(m)})
            elif r == "assistant":
                if _text(m):
                    items.append({"role": "assistant", "content": _text(m)})
                for c in m.get("tool_calls") or []:
                    c = _call(c)
                    items.append({"type": "function_call", "call_id": c.id, "name": c.function,
                                  "arguments": json.dumps(c.args)})
            elif r == "tool":
                out = _text(m)
                if m.get("error"):
                    out = f"Error: {m['error']}"
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": out})
        tools = [{"type": "function", "name": f.name, "description": f.description,
                  "parameters": f.parameters.model_json_schema()} for f in runtime.functions.values()]
        resp = self.client.responses.create(model=self.model, instructions=instructions, input=items,
                                            tools=tools or openai.NOT_GIVEN, reasoning={"effort": self.effort},
                                            store=False)
        text, calls = "", []
        for o in resp.output:
            if o.type == "message":
                text += "".join(getattr(c, "text", "") for c in o.content)
            elif o.type == "function_call":
                calls.append(FunctionCall(function=o.name, args=json.loads(o.arguments) if o.arguments else {},
                                          id=o.call_id))
        out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(text)] if text else None,
                                   tool_calls=calls or None)
        return query, runtime, env, [*messages, out], extra_args
