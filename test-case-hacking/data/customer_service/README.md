# Customer Service data

This directory holds the raw customer problem list for the `customer_service`
dataset, ported from `reward-hacking-evals/datasets/`.

Files:

- `customer_service_problems.txt` — one customer problem description per line
  (tracked; `*.txt` is not gitignored).

To repopulate `customer_service_problems.txt` from the source:

```
cp /path/to/reward-hacking-evals/datasets/customer_service_problems.txt .
```

Or from within this repo's `test-case-hacking/` directory:

```
cp ../../reward-hacking-evals/datasets/customer_service_problems.txt data/customer_service/
```
