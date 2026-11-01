### qwen30b — operating point `native`

Normal tasks (no attack): n=97 (+0 agent errors), undefended utility 72.2% [62.5–80.1].

| Detector | Threshold | Normal-task utility |
|---|---|---|
| secjev_tool | 0.5 | 63.9% [54.0–72.8] |
| secjev_action_meta | 0.5 | 54.6% [44.7–64.2] |
| secjev_action_content | 0.5 | 58.8% [48.8–68.0] |
| piguard | 0.5 | 36.1% [27.2–46.0] |
| deberta | 0.5 | 37.1% [28.2–47.0] |
| llm_judge | 0.5 | 20.6% [13.8–29.7] |

**Attack success rate (lower is better)**

| Attack set | n | Undefended | secjev_tool | secjev_action_meta | secjev_action_content | piguard | deberta | llm_judge |
|---|---|---|---|---|---|---|---|---|
| Standard AgentDojo (in SecJev training) | 949 | 26.4% [23.7–29.3] | 0.0% [0.0–0.4] | 2.7% [1.9–4.0] | 5.5% [4.2–7.1] | 0.3% [0.1–0.9] | 4.7% [3.6–6.3] | 0.0% [0.0–0.4] |
| Bland workflow note (new) | 949 | 9.5% [7.8–11.5] | 0.0% [0.0–0.4] | 0.7% [0.4–1.5] | 2.0% [1.3–3.1] | 0.1% [0.0–0.6] | 2.3% [1.5–3.5] | 0.0% [0.0–0.4] |
| Metadata-field instruction (new) | 949 | 6.2% [4.9–7.9] | 0.2% [0.1–0.8] | 0.5% [0.2–1.2] | 0.2% [0.1–0.8] | 0.0% [0.0–0.4] | 0.9% [0.5–1.8] | 0.1% [0.0–0.6] |
| Quoted earlier reply (new) | 949 | 6.0% [4.7–7.7] | 0.0% [0.0–0.4] | 0.7% [0.4–1.5] | 0.9% [0.5–1.8] | 0.8% [0.4–1.7] | 0.7% [0.4–1.5] | 0.0% [0.0–0.4] |
| Zero-width obfuscation (new) | 949 | 4.1% [3.0–5.6] | 0.0% [0.0–0.4] | 0.2% [0.1–0.8] | 0.7% [0.4–1.5] | 1.6% [1.0–2.6] | 0.0% [0.0–0.4] | 0.0% [0.0–0.4] |
| LLMail human attacks (new) | 800 | 1.1% [0.6–2.1] | 0.2% [0.1–0.9] | 0.0% [0.0–0.5] | 0.1% [0.0–0.7] | 0.0% [0.0–0.5] | 0.4% [0.1–1.1] | 0.0% [0.0–0.5] |

**Task completion under attack (higher is better)**

| Attack set | n | Undefended | secjev_tool | secjev_action_meta | secjev_action_content | piguard | deberta | llm_judge |
|---|---|---|---|---|---|---|---|---|
| Standard AgentDojo (in SecJev training) | 949 | 55.2% [52.0–58.4] | 6.6% [5.2–8.4] | 42.9% [39.8–46.1] | 36.0% [33.0–39.1] | 10.3% [8.5–12.4] | 26.6% [23.8–29.5] | 6.6% [5.2–8.4] |
| Bland workflow note (new) | 949 | 62.6% [59.5–65.6] | 7.3% [5.8–9.1] | 49.9% [46.8–53.1] | 46.4% [43.2–49.5] | 16.2% [14.0–18.7] | 35.2% [32.2–38.3] | 7.7% [6.2–9.6] |
| Metadata-field instruction (new) | 949 | 62.9% [59.8–65.9] | 10.1% [8.4–12.2] | 53.1% [49.9–56.3] | 54.6% [51.4–57.7] | 13.0% [11.0–15.2] | 25.5% [22.8–28.4] | 7.7% [6.2–9.6] |
| Quoted earlier reply (new) | 949 | 61.7% [58.6–64.8] | 11.8% [9.9–14.0] | 51.0% [47.8–54.2] | 53.0% [49.8–56.2] | 17.3% [15.0–19.8] | 26.8% [24.0–29.7] | 10.2% [8.5–12.3] |
| Zero-width obfuscation (new) | 949 | 62.9% [59.8–65.9] | 7.1% [5.6–8.9] | 51.0% [47.8–54.2] | 50.8% [47.6–54.0] | 13.7% [11.7–16.0] | 14.1% [12.0–16.5] | 9.6% [7.9–11.6] |
| LLMail human attacks (new) | 800 | 65.5% [62.1–68.7] | 12.8% [10.6–15.2] | 57.0% [53.5–60.4] | 56.8% [53.3–60.1] | 7.6% [6.0–9.7] | 30.2% [27.2–33.5] | 9.8% [7.9–12.0] |

### qwen30b — operating point `u95`

Normal tasks (no attack): n=97 (+0 agent errors), undefended utility 72.2% [62.5–80.1].

| Detector | Threshold | Normal-task utility |
|---|---|---|
| secjev_tool | 0.89 | 69.1% [59.3–77.4] |
| secjev_action_meta | 0.88 | 69.1% [59.3–77.4] |
| secjev_action_content | 0.72 | 69.1% [59.3–77.4] |
| piguard | n/a | n/a |
| deberta | n/a | n/a |
| llm_judge | n/a | n/a |

**Attack success rate (lower is better)**

| Attack set | n | Undefended | secjev_tool | secjev_action_meta | secjev_action_content | piguard | deberta | llm_judge |
|---|---|---|---|---|---|---|---|---|
| Standard AgentDojo (in SecJev training) | 949 | 26.4% [23.7–29.3] | 0.0% [0.0–0.4] | 17.3% [15.0–19.8] | 10.9% [9.0–13.0] | n/a | n/a | n/a |
| Bland workflow note (new) | 949 | 9.5% [7.8–11.5] | 0.1% [0.0–0.6] | 5.6% [4.3–7.2] | 3.8% [2.8–5.2] | n/a | n/a | n/a |
| Metadata-field instruction (new) | 949 | 6.2% [4.9–7.9] | 0.4% [0.2–1.1] | 3.2% [2.2–4.5] | 2.7% [1.9–4.0] | n/a | n/a | n/a |
| Quoted earlier reply (new) | 949 | 6.0% [4.7–7.7] | 0.2% [0.1–0.8] | 3.2% [2.2–4.5] | 2.6% [1.8–3.9] | n/a | n/a | n/a |
| Zero-width obfuscation (new) | 949 | 4.1% [3.0–5.6] | 0.0% [0.0–0.4] | 2.3% [1.5–3.5] | 1.9% [1.2–3.0] | n/a | n/a | n/a |
| LLMail human attacks (new) | 800 | 1.1% [0.6–2.1] | 0.8% [0.3–1.6] | 0.8% [0.3–1.6] | 0.4% [0.1–1.1] | n/a | n/a | n/a |

**Task completion under attack (higher is better)**

| Attack set | n | Undefended | secjev_tool | secjev_action_meta | secjev_action_content | piguard | deberta | llm_judge |
|---|---|---|---|---|---|---|---|---|
| Standard AgentDojo (in SecJev training) | 949 | 55.2% [52.0–58.4] | 6.6% [5.2–8.4] | 51.5% [48.3–54.7] | 42.9% [39.8–46.1] | n/a | n/a | n/a |
| Bland workflow note (new) | 949 | 62.6% [59.5–65.6] | 7.9% [6.4–9.8] | 60.0% [56.8–63.0] | 53.3% [50.1–56.5] | n/a | n/a | n/a |
| Metadata-field instruction (new) | 949 | 62.9% [59.8–65.9] | 11.0% [9.1–13.1] | 61.1% [58.0–64.2] | 58.8% [55.6–61.9] | n/a | n/a | n/a |
| Quoted earlier reply (new) | 949 | 61.7% [58.6–64.8] | 14.0% [12.0–16.4] | 59.9% [56.7–62.9] | 58.2% [55.0–61.3] | n/a | n/a | n/a |
| Zero-width obfuscation (new) | 949 | 62.9% [59.8–65.9] | 7.5% [6.0–9.3] | 61.2% [58.1–64.3] | 57.4% [54.3–60.5] | n/a | n/a | n/a |
| LLMail human attacks (new) | 800 | 65.5% [62.1–68.7] | 20.8% [18.1–23.7] | 63.6% [60.2–66.9] | 61.6% [58.2–64.9] | n/a | n/a | n/a |

### Adaptive attacker

| Agent | Defense | Pairs | Attacker success within 10 rounds | Mean rounds to success |
|---|---|---|---|---|
| qwen30b | none | 60 (+0 errors) | 26.7% [17.1–39.0] | 3.7 |
| qwen30b | secjev_tool | 60 (+0 errors) | 8.3% [3.6–18.1] | 2.8 |
| qwen30b | secjevtool-u95 | 60 (+0 errors) | 13.3% [6.9–24.2] | 2.6 |
| qwen30b | secjev_action_meta | 60 (+0 errors) | 1.7% [0.3–8.9] | 6.0 |
| qwen30b | secjev_action_content | 60 (+0 errors) | 3.3% [0.9–11.4] | 4.0 |
| qwen30b | llm_judge | 60 (+0 errors) | 0.0% [0.0–6.0] | – |
| qwen30b | piguard | 60 (+0 errors) | 0.0% [0.0–6.0] | – |
| qwen30b | deberta | 60 (+0 errors) | 10.0% [4.7–20.1] | 2.3 |

