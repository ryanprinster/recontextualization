# Data Entry data

Raw JSONL task files for the `data_entry` dataset, ported from
`reward-hacking-evals/datasets/data_entry/`.

Each JSONL line describes a single task (a math problem, MMLU question, APPS
problem, or xsum story). A `DataEntrySample` groups multiple tasks into one
agent trajectory and writes them to `/mnt/tasks/<label>.txt` inside the
sandbox.

Files (all `*.jsonl` are gitignored, so none are tracked):

- `math_sample.jsonl` — MATH problems with integer gold answers. Default.
- `mmlu_sample.jsonl` — MMLU multiple-choice questions; gold is an (A)-(D) tag.
- `apps_introductory.jsonl` / `apps_interview.jsonl` / `apps_competition.jsonl`
  — APPS Python coding problems (used as `apps_easy` / `apps_hard`).
- `xsum_stories_sample.jsonl` — XSum articles with gold summaries.

To populate, copy from the original repo:

```
cp /workspace/reward-hacking-evals/datasets/data_entry/math_sample.jsonl \
   data/data_entry/
```

Per-problem-type file conventions:

| problem_type   | source JSONL               | text field | label field | expected field |
|----------------|----------------------------|------------|-------------|----------------|
| math           | math_sample.jsonl          | `text`     | `label`     | `answer` (int) |
| mmlu           | mmlu_sample.jsonl          | `text`     | `label`     | `answer` (int index 0..3) |
| apps_easy      | apps_introductory.jsonl    | `text`     | `label`     | — (no gold solution) |
| apps_hard      | apps_competition.jsonl     | `text`     | `label`     | — (no gold solution) |
| xsum_stories   | xsum_stories_sample.jsonl  | `text`     | `label`     | `summary` (str) |
