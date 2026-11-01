"""Generate identifier-checked paraphrases of AgentDojo user prompts before scored runs.

Lexical checks are conservative safeguards, not proof of semantic equivalence. Freeze and review
accepted prompts before evaluation; paraphrases of a task must remain in the same suite split.
"""
import argparse, hashlib, json, os, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Order matters: capture structured identifiers and ordinals before generic numbers/words.
STRUCTURED = (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|https?://[^\s<>\"']+|"
              r"www\.[\w-]+(?:\.[\w-]+)+|\b[\w-]+(?:\.[\w-]+)+\b|"
              r"\b[A-Z]{2}\d{2}[A-Z0-9]{8,30}\b|\b\d+(?:st|nd|rd|th)\b|"
              r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b|\b\d[\d,.:/-]*\b")
TOKEN = re.compile(r"(?<![A-Za-z])'[^'\n]+'(?![A-Za-z])|\"[^\"\n]+\"|" + STRUCTURED + r"|\b[A-Z][a-zA-Z]+\b")
STRUCTURED_TOKEN = re.compile(STRUCTURED)
# Only generic request openers may be ignored at sentence starts. Names such as Alice must survive.
OPENERS = {"Please", "Can", "Could", "Would", "Will", "What", "When", "Where", "Which", "Who", "How", "Why",
           "The", "This", "That", "These", "Those", "My", "We", "Our", "You", "Your", "It", "If", "For", "In",
           "On", "At", "By", "Also", "Then", "And", "Hi", "Hello", "Hey", "Thanks", "Thank", "Do", "Tell",
           "Find", "Show", "Give", "Get", "Read", "Send", "Summarize", "Check", "Make", "Use", "Book", "Add"}
PROMPT = ("Rewrite the following request {k} different ways, as different people might naturally phrase it. Keep the "
          "meaning exactly the same and keep every name, number, date, amount, email address, account number, URL, "
          "file name and quoted text EXACTLY as written. Do not add or drop any detail. Return a JSON list of {k} "
          "strings and nothing else.\n\nRequest:\n{p}")


def _tokens(prompt):
    for match in TOKEN.finditer(prompt):
        token = match.group(0)
        if token[:1] not in ("'", '"'):
            token = token.rstrip(".,;:!?)")
        before = prompt[:match.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:"
        if token and not (sentence_start and token in OPENERS):
            yield token


def required(p):
    """Protected quoted text, names, numbers (including ordinals), and structured identifiers."""
    return set(_tokens(p))


def _normalize(text):
    return " ".join(text.split())


def validate_paraphrase(original, candidate):
    """Conservative lexical validation; no claim of semantic equivalence."""
    if not isinstance(candidate, str) or not candidate.strip() or _normalize(candidate) == _normalize(original):
        return False
    for token, count in Counter(_tokens(original)).items():
        pattern = (r"(?<!\w)" if token[0].isalnum() else "") + re.escape(token)
        pattern += r"(?!\w)" if token[-1].isalnum() else ""
        if len(re.findall(pattern, candidate)) < count:
            return False
    # Reject changed/added numeric or structured details, not just missing originals.
    def structured(text):
        return Counter(m.group(0).rstrip(".,;:!?)") for m in STRUCTURED_TOKEN.finditer(text))
    return structured(candidate) == structured(original)


def parse_candidates(text):
    """Accept a JSON string array (optionally code-fenced), never dict keys or scalar strings."""
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S | re.I)
        if not match:
            return []
        text = match.group(1)
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        return []
    return values if isinstance(values, list) and all(isinstance(v, str) for v in values) else []


def accept_paraphrases(original, candidates):
    accepted, seen = [], set()
    for value in candidates:
        if not validate_paraphrase(original, value):
            continue
        key = _normalize(value)
        if key not in seen:
            accepted.append(value.strip())
            seen.add(key)
    return accepted


def main():
    import openai
    from agentdojo.task_suite.load_suites import get_suites

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.n < 1:
        ap.error("--n must be positive")
    c = openai.OpenAI(base_url=a.base_url, api_key=os.environ.get("OPENAI_API_KEY", "local"), timeout=600, max_retries=0)
    suites = get_suites("v1.2.2")
    tasks = {(s, tid): ut for s in ("workspace", "travel", "banking", "slack")
             for tid, ut in suites[s].user_tasks.items()}
    source_hash = {key: hashlib.sha256(ut.PROMPT.encode("utf-8")).hexdigest() for key, ut in tasks.items()}

    def one(item):
        (s, tid), ut = item
        got = []
        for attempt in range(8):
            r = c.chat.completions.create(model=a.model, temperature=0.7,
                                          messages=[{"role": "user", "content": PROMPT.format(k=a.n, p=ut.PROMPT)}])
            got = accept_paraphrases(ut.PROMPT, [*got, *parse_candidates(r.choices[0].message.content or "")])
            if len(got) >= a.n:
                return s, tid, got[:a.n]
        raise RuntimeError(f"{s}/{tid}: only {len(got)} identifier-checked paraphrases (needs {sorted(required(ut.PROMPT))})")

    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    prog = Path(a.out + ".progress.jsonl")
    done = {}
    if prog.exists():
        with prog.open() as f:
            for line in f:
                row = json.loads(line)
                key = (row["suite"], row["task"])
                if key not in tasks or row.get("source_sha256") != source_hash[key]:
                    continue  # Never reuse a paraphrase of a different/unknown source prompt.
                values = row.get("paraphrases")
                checked = accept_paraphrases(tasks[key].PROMPT, values) if isinstance(values, list) else []
                if len(checked) >= a.n:
                    done[key] = checked[:a.n]
    todo = [item for item in tasks.items() if item[0] not in done]

    def safe(item):
        try:
            return one(item), None
        except RuntimeError as e:
            return None, str(e)
    failures = []
    with ThreadPoolExecutor(32) as ex, prog.open("a") as f:
        for res, err in ex.map(safe, todo):
            if err:
                failures.append(err)
                continue
            s, tid, ps = res
            done[(s, tid)] = ps
            f.write(json.dumps({"suite": s, "task": tid, "source_sha256": source_hash[(s, tid)],
                                "paraphrases": ps}, ensure_ascii=False) + "\n")
            f.flush()
    if failures:
        raise SystemExit("paraphrase failures (rerun to retry only these):\n" + "\n".join(failures))
    out = {}
    for (s, tid), ps in done.items():
        out.setdefault(s, {})[tid] = ps
    output.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("paraphrased", len(done), "tasks; identifier checks do not replace semantic review")


if __name__ == "__main__":
    main()
