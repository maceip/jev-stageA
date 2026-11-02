# E2E test results (real AgentDojo, real SecJev, real OpenAI; no mocks)

Commit under test: e73a04c. Suite: tests/ (19 tests).

| Where | Hardware | Scope | Result |
|---|---|---|---|
| jev-score2-h100 (Lambda) | 1x H100 80 GB | full suite | 19 passed (195.7 s) |
| jev-secjev-a100 (Lambda) | 1x A100 40 GB | full suite | 19 passed (211.6 s) |
| Claude workspace | 2 CPU, 7 GB, no GPU | the 7 tests that need no SecJev | 7 passed (29.8 s) |

The workspace cannot hold SecJev-4B (no GPU, 7 GB RAM); the 12 GPU tests run on the servers only.
Fixed on the way here: nonexistent default model gpt-6.1-astra (404), error-field reads in the action monitor,
GPU OOM on 40 GB (models were held across modules), mock-based tests/test_jev.py removed.
