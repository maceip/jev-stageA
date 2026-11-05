---
configs:
- config_name: defense
  data_files:
  - split: valid
    path: valid_defense.json
- config_name: interaction_chats
  data_files:
  - split: attack
    path: chat.json
- config_name: message_counts
  data_files:
  - split: defense_teams
    path: message_counts.csv
- config_name: teams
  data_files:
  - split: defense_teams
    path: teams.json
license: mit
---

# Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition

Welcome to the dataset of the [LLM CTF @ SaTML 2024](https://ctf.spylab.ai/). You can find a detailed report of the competition [here](https://arxiv.org/abs/2406.07954).

The task of the attacker was to discover a secret from the start of the chat, hidden by the defending team's system prompt, Python filters, and LLM filters.
The models used were `gpt-3.5-turbo-1106` and `llama-2-70b-chat`.

The dataset contains three splits:
- `teams`: mapping between team names and the ids of the submitted defenses.
- `interaction_chats`: contains 137k adversarial chats collected during the competition, together with the defenses and secret guesses.
- `defense`: includes all 44 accepted defenses from the [leaderboard](https://ctf.spylab.ai/leaderboard)

For more details on the schema you can check the [official report](https://arxiv.org/abs/2406.07954).

All defenses in `valid_defense.json` passed the utility evaluation (MMLU-like benchmarks + subjective chat eval of pre-specified type conducted by the organizers) and were deemed not to diminish the quality of the model in non-adversarial settings.

## Usage

To load the data using the `datasets` library you can run

```
from datasets import load_dataset

defenses = load_dataset("ethz-spylab/ctf-satml24", "defense")["valid"]

teams = load_dataset("ethz-spylab/ctf-satml24", "teams")["defense_teams"]

chats = load_dataset("ethz-spylab/ctf-satml24", "interaction_chats")["attack"]
```

[Our repository](https://github.com/ethz-spylab/satml-llm-ctf) contains tools to reproduce the data analysis performed in our official report.

## License
```
MIT License

Copyright (c) 2024 all participants of the LLM CTF @ SaTML 2024, Edoardo Debenedetti, Javier Rando, Daniel Paleka

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```