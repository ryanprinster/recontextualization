# Sales Environment data

This directory holds the pre-defined customer email threads for the
`sales_environment` dataset, ported from
`reward-hacking-evals/sales_env_experiments/prompts.py`.

Files:

- `threads.jsonl` — one JSON object per email thread (not tracked; `*.jsonl`
  is gitignored). Each object has:
  - `thread_id` (str), `status` ("valid" | "invalid" | "no_sale"),
  - `expected_sale_amount` (int), `violation_type` (str | null),
  - `customer` (str),
  - `emails` (list of `{timestamp, from, to, [cc,] subject, body}`).

## How to populate

From the repo root (with the reward-hacking-evals checkout adjacent):

```
python - <<'PY'
import json, sys
sys.path.insert(0, "../reward-hacking-evals/sales_env_experiments")
from prompts import email_threads
with open("data/sales_environment/threads.jsonl", "w") as f:
    for t in email_threads:
        f.write(json.dumps(t, ensure_ascii=True) + "\n")
PY
```

Or, if running from this repo's `test-case-hacking/` directory:

```
cd /path/to/reward-hacking-evals/sales_env_experiments
python -c "
import json
from prompts import email_threads
with open('/path/to/recontextualization/test-case-hacking/data/sales_environment/threads.jsonl', 'w') as f:
    for t in email_threads:
        f.write(json.dumps(t, ensure_ascii=True) + '\n')
"
```

The source file defines 8 threads: 1 valid sale, 4 invalid (one per violation
type — excessive_discount, discontinued_product, wrong_territory
/ lead_attribution_theft, premature_confirmation), and 3 no-response threads.
