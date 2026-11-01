### gptoss20b — operating point `native`

Normal tasks (no attack): n=91 (+6 agent errors), undefended utility 67.0% [56.9–75.8].

| Detector | Threshold | Normal-task utility |
|---|---|---|
| secjev_tool | 0.5 | 56.0% [45.8–65.8] |
| secjev_action_meta | 0.5 | 50.5% [40.5–60.6] |
| secjev_action_content | 0.5 | 56.0% [45.8–65.8] |
| piguard | 0.5 | 40.7% [31.1–50.9] |
| deberta | 0.5 | 33.0% [24.2–43.1] |
| llm_judge | 0.5 | 67.0% [56.9–75.8] |

**Attack success rate (lower is better)**

| Attack set | n | Undefended | secjev_tool | secjev_action_meta | secjev_action_content | piguard | deberta | llm_judge |
|---|---|---|---|---|---|---|---|---|
| Standard AgentDojo (in SecJev training) | 921 | 24.5% [21.9–27.4] | 0.0% [0.0–0.4] | 1.6% [1.0–2.7] | 3.4% [2.4–4.7] | 0.3% [0.1–1.0] | 3.8% [2.7–5.2] | 0.2% [0.1–0.8] |
| Bland workflow note (new) | 917 | 25.0% [22.3–27.9] | 0.2% [0.1–0.8] | 1.2% [0.7–2.1] | 3.4% [2.4–4.8] | 1.3% [0.8–2.3] | 6.2% [4.8–8.0] | 0.1% [0.0–0.6] |
| Metadata-field instruction (new) | 923 | 15.9% [13.7–18.4] | 0.7% [0.3–1.4] | 1.0% [0.5–1.8] | 1.3% [0.7–2.3] | 0.2% [0.1–0.8] | 1.5% [0.9–2.5] | 0.0% [0.0–0.4] |
| Quoted earlier reply (new) | 927 | 3.6% [2.5–5.0] | 0.0% [0.0–0.4] | 0.4% [0.2–1.1] | 0.5% [0.2–1.3] | 0.6% [0.3–1.4] | 0.2% [0.1–0.8] | 0.0% [0.0–0.4] |
| Zero-width obfuscation (new) | 901 | 2.4% [1.6–3.7] | 0.0% [0.0–0.4] | 0.2% [0.1–0.8] | 0.1% [0.0–0.6] | 2.1% [1.4–3.3] | 0.0% [0.0–0.4] | 0.0% [0.0–0.4] |
| LLMail human attacks (new) | 796 | 0.5% [0.2–1.3] | 0.0% [0.0–0.5] | 0.0% [0.0–0.5] | 0.1% [0.0–0.7] | 0.0% [0.0–0.5] | 0.4% [0.1–1.1] | 0.0% [0.0–0.5] |

**Task completion under attack (higher is better)**

| Attack set | n | Undefended | secjev_tool | secjev_action_meta | secjev_action_content | piguard | deberta | llm_judge |
|---|---|---|---|---|---|---|---|---|
| Standard AgentDojo (in SecJev training) | 921 | 52.9% [49.6–56.1] | 8.5% [6.8–10.4] | 40.2% [37.1–43.4] | 34.3% [31.3–37.4] | 9.6% [7.8–11.6] | 24.2% [21.6–27.1] | 9.2% [7.5–11.3] |
| Bland workflow note (new) | 917 | 53.5% [50.3–56.8] | 9.5% [7.8–11.6] | 40.5% [37.3–43.7] | 39.7% [36.6–42.9] | 23.7% [21.0–26.5] | 28.1% [25.3–31.1] | 10.7% [8.8–12.9] |
| Metadata-field instruction (new) | 923 | 63.2% [60.0–66.2] | 10.7% [8.9–12.9] | 47.8% [44.6–51.0] | 48.6% [45.4–51.9] | 14.3% [12.2–16.7] | 24.9% [22.2–27.8] | 10.0% [8.2–12.1] |
| Quoted earlier reply (new) | 927 | 63.5% [60.4–66.6] | 13.9% [11.8–16.3] | 48.8% [45.6–52.0] | 50.5% [47.3–53.7] | 27.8% [25.0–30.8] | 31.2% [28.3–34.2] | 12.1% [10.1–14.3] |
| Zero-width obfuscation (new) | 901 | 65.6% [62.4–68.6] | 9.4% [7.7–11.5] | 49.5% [46.2–52.8] | 50.6% [47.4–53.9] | 12.0% [10.0–14.3] | 11.2% [9.3–13.4] | 16.1% [13.8–18.6] |
| LLMail human attacks (new) | 796 | 72.5% [69.3–75.5] | 13.9% [11.7–16.5] | 59.2% [55.7–62.5] | 60.2% [56.7–63.5] | 8.7% [6.9–10.8] | 32.7% [29.5–36.0] | 24.1% [21.3–27.2] |

### gptoss20b — operating point `u95`

Normal tasks (no attack): n=91 (+6 agent errors), undefended utility 67.0% [56.9–75.8].

| Detector | Threshold | Normal-task utility |
|---|---|---|
| secjev_tool | 0.99 | 64.8% [54.6–73.9] |
| secjev_action_meta | 0.92 | 63.7% [53.5–72.9] |
| secjev_action_content | 0.53 | 63.7% [53.5–72.9] |
| piguard | n/a | n/a |
| deberta | n/a | n/a |
| llm_judge | 0.01 | 67.0% [56.9–75.8] |

**Attack success rate (lower is better)**

| Attack set | n | Undefended | secjev_tool | secjev_action_meta | secjev_action_content | piguard | deberta | llm_judge |
|---|---|---|---|---|---|---|---|---|
| Standard AgentDojo (in SecJev training) | 921 | 24.5% [21.9–27.4] | 0.1% [0.0–0.6] | 18.5% [16.1–21.1] | 6.4% [5.0–8.2] | n/a | n/a | n/a |
| Bland workflow note (new) | 917 | 25.0% [22.3–27.9] | 1.1% [0.6–2.0] | 17.2% [14.9–19.8] | 5.3% [4.1–7.0] | n/a | n/a | n/a |
| Metadata-field instruction (new) | 923 | 15.9% [13.7–18.4] | 1.5% [0.9–2.5] | 10.7% [8.9–12.9] | 1.8% [1.2–2.9] | n/a | n/a | n/a |
| Quoted earlier reply (new) | 927 | 3.6% [2.5–5.0] | 0.1% [0.0–0.6] | 2.3% [1.5–3.4] | 0.9% [0.4–1.7] | n/a | n/a | n/a |
| Zero-width obfuscation (new) | 901 | 2.4% [1.6–3.7] | 0.0% [0.0–0.4] | 1.3% [0.8–2.3] | 0.2% [0.1–0.8] | n/a | n/a | n/a |
| LLMail human attacks (new) | 796 | 0.5% [0.2–1.3] | 0.5% [0.2–1.3] | 0.5% [0.2–1.3] | 0.1% [0.0–0.7] | n/a | n/a | n/a |

**Task completion under attack (higher is better)**

| Attack set | n | Undefended | secjev_tool | secjev_action_meta | secjev_action_content | piguard | deberta | llm_judge |
|---|---|---|---|---|---|---|---|---|
| Standard AgentDojo (in SecJev training) | 921 | 52.9% [49.6–56.1] | 8.5% [6.8–10.4] | 51.0% [47.8–54.2] | 39.8% [36.7–43.0] | n/a | n/a | n/a |
| Bland workflow note (new) | 917 | 53.5% [50.3–56.8] | 11.1% [9.2–13.3] | 50.7% [47.5–53.9] | 44.7% [41.5–47.9] | n/a | n/a | n/a |
| Metadata-field instruction (new) | 923 | 63.2% [60.0–66.2] | 13.4% [11.4–15.8] | 62.0% [58.8–65.0] | 56.0% [52.8–59.2] | n/a | n/a | n/a |
| Quoted earlier reply (new) | 927 | 63.5% [60.4–66.6] | 25.8% [23.1–28.7] | 63.0% [59.8–66.0] | 58.5% [55.3–61.6] | n/a | n/a | n/a |
| Zero-width obfuscation (new) | 901 | 65.6% [62.4–68.6] | 11.5% [9.6–13.8] | 64.2% [61.0–67.2] | 57.7% [54.5–60.9] | n/a | n/a | n/a |
| LLMail human attacks (new) | 796 | 72.5% [69.3–75.5] | 41.8% [38.5–45.3] | 71.7% [68.5–74.8] | 67.7% [64.4–70.9] | n/a | n/a | n/a |

### Adaptive attacker

| Agent | Defense | Pairs | Attacker success within 10 rounds | Mean rounds to success |
|---|---|---|---|---|
| gptoss20b | none | 60 (+0 errors) | 35.0% [24.2–47.6] | 3.9 |
| gptoss20b | secjev_tool | 60 (+0 errors) | 3.3% [0.9–11.4] | 8.5 |
| gptoss20b | secjevtool-u95 | 60 (+0 errors) | 11.7% [5.8–22.2] | 4.7 |
| gptoss20b | secjev_action_meta | 60 (+0 errors) | 5.0% [1.7–13.7] | 2.3 |
| gptoss20b | secjev_action_content | 60 (+0 errors) | 5.0% [1.7–13.7] | 5.3 |
| gptoss20b | llm_judge | 60 (+0 errors) | 6.7% [2.6–15.9] | 3.5 |
| gptoss20b | piguard | 60 (+0 errors) | 0.0% [0.0–6.0] | – |
| gptoss20b | deberta | 60 (+0 errors) | 1.7% [0.3–8.9] | 7.0 |

