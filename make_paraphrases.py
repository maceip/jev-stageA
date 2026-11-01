"""Generate 4 verified paraphrases of every AgentDojo user-task prompt (frozen before scored runs).

A paraphrase is accepted only if every identifier-like token of the original (numbers, dates, emails, IBANs,
URLs, quoted strings, capitalised words) appears verbatim in it. Fails if a task cannot get 4 accepted.
"""
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor

import openai
from agentdojo.task_suite.load_suites import get_suites

TOKEN = re.compile(
    r"(?:(?<=^)|(?<=[\s(\[{]))'[^'\n]+'(?=$|[\s.,;:!?)\]}])"
    r'|\"[^\"\n]+\"'
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|https?://[^\s]+"
    r"|www\.[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}"
    r"|\b[\w-]+\.[A-Za-z][A-Za-z0-9]{1,4}\b"
    r"|\b[A-Z]{2}\d{2}[A-Z0-9]{8,30}\b"
    r"|\b\d+(?:[.,:/-]\d+)*\b"
    r"|\b[A-Z][a-zA-Z]+\b"
)
PROMPT = ("Rewrite the following request {k} different ways, as different people might naturally phrase it. Keep the "
          "meaning exactly the same and keep every name, number, date, amount, email address, account number, URL, "
          "file name and quoted text EXACTLY as written. Do not add or drop any detail. Return a JSON list of {k} "
          "strings and nothing else.\n\nRequest:\n{p}")


def required(p):
    """Identifier-like tokens; a capitalised word that only ever appears at a sentence/clause start is not one."""
    need = set()
    clause_starts = set()
    non_clause = set()
    for m in TOKEN.finditer(p):
        t = m.group(0).strip(".,;:!?)('\"").strip()
        if not t:
            continue
        before = p[:m.start()].rstrip()
        is_start = before == "" or before[-1] in ".!?:\n;-—(\"'"
        if re.fullmatch(r"[A-Z][a-z]+", t):
            if is_start:
                clause_starts.add(t)
            else:
                non_clause.add(t)
        else:
            need.add(t)
    return need | non_clause


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="gpt-6.1-astra")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import os
    c = openai.OpenAI(base_url=a.base_url, api_key=os.environ.get("OPENAI_API_KEY", "local"), timeout=600, max_retries=0)
    suites = get_suites("v1.2.2")
    tasks = [(s, ut) for s in ("workspace", "travel", "banking", "slack") for ut in suites[s].user_tasks.values()]

    def one(item):
        s, ut = item
        need, got = required(ut.PROMPT), []
        for attempt in range(8):
            r = c.chat.completions.create(model=a.model, temperature=0.7,
                                          messages=[{"role": "user", "content": PROMPT.format(k=a.n, p=ut.PROMPT)}])
            txt = (r.choices[0].message.content or "").strip()
            m = re.search(r"\[.*\]", txt, re.S)
            try:
                cand = json.loads(m.group(0)) if m else []
            except json.JSONDecodeError:
                cand = []
            for x in cand:
                if isinstance(x, str) and x.strip() != ut.PROMPT.strip() and all(t in x for t in need) and x not in got:
                    got.append(x)
            if len(got) >= a.n:
                return s, ut.ID, got[: a.n]
        raise RuntimeError(f"{s}/{ut.ID}: only {len(got)} verified paraphrases (needs {sorted(need)})")

    import os
    prog = a.out + ".progress.jsonl"
    done = {}
    if os.path.exists(prog):
        for l in open(prog):
            r = json.loads(l); done[(r["suite"], r["task"])] = r["paraphrases"]
    todo = [t for t in tasks if (t[0], t[1].ID) not in done]

    def safe(item):
        try:
            return one(item), None
        except RuntimeError as e:
            return None, str(e)
    failures = []
    with ThreadPoolExecutor(32) as ex, open(prog, "a") as f:
        for res, err in ex.map(safe, todo):
            if err:
                failures.append(err); continue
            s, tid, ps = res
            done[(s, tid)] = ps
            f.write(json.dumps({"suite": s, "task": tid, "paraphrases": ps}, ensure_ascii=False) + "\n"); f.flush()
    if failures:
        raise SystemExit("paraphrase failures (rerun to retry only these):\n" + "\n".join(failures))
    out = {}
    for (s, tid), ps in done.items():
        out.setdefault(s, {})[tid] = ps
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("paraphrased", len(done), "tasks")


if __name__ == "__main__":
    main()
