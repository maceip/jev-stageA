# Security policy

jev-stageA is security research software: the repository contains prompt-injection payloads, an adaptive attacker, and experimental agent defenses. The defenses carry no production security guarantee; the pre-registered evaluation gate for a production build was not passed (see the README results section).

## Reporting a vulnerability

Report vulnerabilities in the defense code, the checkpoint loading path, or the evaluation harness through GitHub private vulnerability reporting on the jev-stageA repository. Include the affected file, a minimal reproduction against AgentDojo, and the checkpoint and model revisions used. Do not open a public issue for a vulnerability that affects systems beyond the benchmark environment.

## Scope notes

- Successful prompt injections against the benchmark defenses are expected research outcomes, not vulnerabilities. Report novel bypasses as evaluation contributions (see CONTRIBUTING.md) unless a bypass also breaks a stated fail-closed invariant in `defenses.py`.
- Use the attack code only against systems you own or are authorized to test.
