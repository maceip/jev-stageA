"""Content-hash deduplication of decision points, within and across attack runs.

Every attack set in Stage A replays the *same* user tasks against the *same*
AgentDojo environment, changing only the injected text. A trajectory therefore
diverges from the benign one only after the first tool result that carries a
payload. Everything before that point - and, for a refused injection, most of
what follows - is byte-identical input to the scorer, yet score.py scores it
again for every one of the six attack sets.

This module keys a decision point on exactly the bytes the scorer reads, so
identical points are scored once. Two consequences:

  * wall-clock: the duplicated prefix is scored 1x instead of 6x;
  * determinism: a repeated run reproduces the same score for the same point,
    so partial results can be resumed without rescoring.

The key deliberately includes the untrusted text a scorer sees (the tool
output, or the untrusted outputs for the action monitor). Two action points
that differ only in provenance-free history are the same point; two that differ
in what the model has read are not.
"""
import hashlib
import json
import os
from pathlib import Path


def _stable(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)


def point_key(kind: str, p: dict) -> str:
    """SHA-256 over everything a scorer reads at this decision point."""
    call = p.get("call") or {}
    if kind == "tool":
        parts = ("tool", p.get("user", ""), call.get("function", ""),
                 _stable(call.get("args")), p.get("output", ""))
    else:
        parts = ("action", p.get("user", ""), _stable(call),
                 _stable(p.get("history")),
                 _stable([s.get("output", "") for s in p.get("seen", [])]))
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class ScoreCache:
    """A persisted {point_key: score-dict} memo table.

    Scores are JSON so a cache can be shipped next to the runs and reused by
    analyze.py without touching a GPU. `hits` / `misses` make the speedup
    measurable rather than assumed.
    """

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path else None
        self.table: dict[str, dict] = {}
        self.hits = self.misses = 0
        if self.path is not None and self.path.exists():
            self.load()

    def load(self) -> "ScoreCache":
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.table[row["key"]] = row["score"]
        return self

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for key, score in self.table.items():
                handle.write(json.dumps({"key": key, "score": score}) + "\n")

    def get(self, key: str):
        if key in self.table:
            self.hits += 1
            return self.table[key]
        self.misses += 1
        return None

    def put(self, key: str, score: dict) -> dict:
        self.table[key] = score
        return score

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "reuse_rate": round(self.hits / total, 4) if total else 0.0}


def deduplicated(points: list[tuple], cache: ScoreCache) -> tuple[list, list, dict]:
    """Split decision points into (resolved, todo, index_map).

    `points` is a list of (index, kind, payload). `resolved` is [(index, score)]
    served from the cache; `todo` holds ONE entry per distinct key that still
    needs a model call; `index_map` maps key -> every index sharing it, so a
    single score can be written back to all of them (duplicates inside one
    episode are as common as duplicates across attack sets).
    """
    resolved, todo, index_map = [], [], {}
    for index, kind, payload in points:
        key = point_key(kind, payload)
        index_map.setdefault(key, []).append(index)
        hit = cache.get(key)
        if hit is not None:
            resolved.append((index, hit))
        elif all(t[1] != key for t in todo):
            todo.append((index, key, kind, payload))
    return resolved, todo, index_map
