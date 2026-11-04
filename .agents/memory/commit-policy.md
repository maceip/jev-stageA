---
type: feedback
description: Always commit completed work immediately in jev-stageA
---

Rule for every agent working in jev-stageA: commit completed work immediately, every time. Never leave finished edits uncommitted in a worktree. A task is not done until the work is committed on the branch that owns the worktree. Never lose anyone's work: when uncommitted work from another worker is found in a tree, commit the work as found (snapshot commit) rather than leaving the work dirty. Never force-push. Never delete files to make a tree look clean. .env, checkpoints/, and *.safetensors stay git-ignored and are never committed.