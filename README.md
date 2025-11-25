## 🎭 Erotic/RolePlay Leaderboard

Automated leaderboard evaluating LLMs on roleplay.

🔗 [Leaderboard](https://PsychoSmiley.github.io/RP-Leaderboard/) | [➕ Request Model](../../issues/new?template=model-request.yml) | <details style="display:inline"><summary style="display:inline;cursor:pointer">📦 Run Locally</summary>

```bash
pip install requests
python benchmark.py --model "meta-llama/llama-4-maverick:free" --endpoint https://openrouter.ai/api/v1
```

**Arguments:** `--model <name>` `--endpoint <url>` `--judgeModel <name>` `--judgeEndpoint <url>`

**Fork:** Fork repo → Add secrets: `OPENROUTER_API_KEY`, `OPENAI_API_BASE`, `JUDGE_ENDPOINT` [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) → Enable Actions + Pages

**Rate limits:** 50 req/day free (1,000 with $10+ credit) | Free models: 20 RPM (3s delays)

**Browse models:** [openrouter.ai/models](https://openrouter.ai/models) | **Paid examples (per M tokens):** Claude Sonnet 4.5 ($3/$15), GPT-5 ($1.25/$10), DeepSeek V3.1 ($0.20/$0.80)

</details>

---

## 📊 Methodology

Judge by `anthropic/claude-sonnet-4.5` on total of 325 points: 25 Yes/No (temp 0.0, realism, uncensored, logic, instruction following) + 3 Open-Ended scored 0-100 (temp 0.7, creative storytelling, character depth, worldbuilding, dialogue, narrative, immersion)

- Why LLM-as-judge works? Like agents, evaluating a story is easier than writing it, same as for humans.
- To evaluate E/RP we need to understand what defines it and know common issues to set specific criteria, but it isn't academic:
  - ✍️ Scope: E/RP scores turn-based POV responses vs EQ-Bench, where the focus is third-person stories.
  - 🔎 Targets: to found popular community models (e.g., Mythomax, still subject to preference/style/length biases-won't guarantee your taste).
  - ⚠️ Limitations: no multi-turn or long-term recall for simplicity, but open questions still lean toward the first message of a character instead of a full story; some models may exploit scoring criteria; Judges as jailbreaks for NSFW evaluation are not used, as they may still refuse and be biased; Models refusing open-end questions are receiving worse scores..
