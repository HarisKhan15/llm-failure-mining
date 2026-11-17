# Verification

All checks below were actually run on 4 November 2026 in a Linux environment with
Python 3.13, pandas 3.0, scikit-learn 1.9 and mlxtend 0.25. They were not run on
Windows or macOS.

## Requirement checklist

| ID | Requirement | Type | Acceptance criterion | Evidence | Status |
|---|---|---|---|---|---|
| R1 | Reproducible data download | Essential | File matches pinned SHA-256 | `scripts/fetch_aiw_data.py` downloaded 45 MB and printed "Saved and verified"; second run printed "already present and verified" | Done |
| R2 | Validated loading | Essential | Invalid records dropped and counted | `tests/test_data.py` (7 tests); full data: 0 dropped of 49,838 | Done |
| R3 | Prompt, model and response features | Essential | Correct values for known cases | `tests/test_features.py` (25 tests incl. parametrised cases) | Done |
| R4 | Interpretable error taxonomy | Essential | Every failure gets one category | Shares in `results/summary.json` sum to 1 (checked in end-to-end test) | Done |
| R5 | Clustering of failed responses | Essential | k chosen by silhouette, summary consistent | `test_clustering_returns_consistent_summary`; full run: k = 10, silhouette 0.162 | Done |
| R6 | Association rules | Essential | Only target consequents, lift > 1 | `test_rules_have_single_target_consequent_and_lift_above_one`; full run: 25 rules | Done |
| R7 | Leakage-safe failure prediction | Essential | No response features; no shared groups | `test_prediction_features_exclude_response_information`, `test_grouped_split_never_shares_prompts` | Done |
| R8 | Baselines | Essential | Constant baseline = 0.5 ROC-AUC; priors reported | `test_evaluation_runs_and_constant_baseline_is_chance`; `results/prediction_summary.csv` | Done |
| R9 | Auto-generated report and figures | Essential | All output files created | `test_end_to_end_on_sample` checks 11 files | Done |
| R10 | Clear error messages | Essential | Missing data gives instructions and exit code 1 | `test_cli_reports_missing_data`; manual run shown below | Done |
| R11 | Optional Ollama collector | Optional | Correct request and output schema | `test_collect_with_mocked_ollama` (mocked HTTP only) | Partly verified |
| R12 | Documentation | Essential | README, explanation, demo, verification, notices | Files in project root | Done |

## Commands run and results

```text
$ python3 -m pytest -q
46 passed in 8.69s

$ python3 -m llm_failure_mining.cli run --sample
...
Done in 4.8 s. Report: .../results/sample/REPORT.md

$ python3 scripts/fetch_aiw_data.py
Downloading https://raw.githubusercontent.com/LAION-AI/AIW/4d5d079b773ae6ab7d2eb852971ea3fda0ea1ce8/collected_responses/AIW_AIW_plus.json
(about 45 MB, this can take a minute)
Saved and verified: .../data/raw/AIW_AIW_plus.json

$ python3 -m llm_failure_mining.cli run
1/6 Loading .../data/raw/AIW_AIW_plus.json
    49838 responses, 75 models, 63 prompts
...
    best k = 10
...
    25 rules kept
...
Done in 37.5 s. Report: .../results/REPORT.md

$ python3 -m llm_failure_mining.cli run --data /nonexistent.json
Error: Dataset not found at /nonexistent.json. Run 'python scripts/fetch_aiw_data.py' or use the sample with '--sample'.
exit=1
```

## Mocked versus real tests

- **Real data:** loading, features, clustering, rules and prediction tests use the committed sample of 435 real AIW records. The full pipeline was run on all 49,838 records.
- **Mocked:** the Ollama collector test replaces the HTTP call with a fake response. It verifies request content, answer parsing and output format, but not a real model.

## Important results (full data)

| Result | Value | Source |
|---|---|---|
| Overall accuracy | 7.2% | `summary.json` |
| Failures that are right − 1 | 58.0% (68.8% on sibling problems) | `summary.json` |
| Accuracy vs off-by-one share, 49 models | Spearman ρ = −0.058 | `summary.json` |
| Best cluster count | k = 10, silhouette 0.162 | `REPORT.md` |
| Unseen prompts, prior_by_model | ROC-AUC 0.837 ± 0.055, PR-AUC 0.314 ± 0.059 | `prediction_summary.csv` |
| Unseen prompts, gradient boosting with model id | ROC-AUC 0.845 ± 0.028, PR-AUC 0.453 ± 0.090 | `prediction_summary.csv` |
| Paired comparison vs prior | ROC-AUC better in 3/5 folds (+0.008), PR-AUC better in 4/5 folds (+0.140) | `REPORT.md` |
| Unseen models, best method | prior_by_family ROC-AUC 0.676 ± 0.093 | `prediction_summary.csv` |

## Unresolved problems and limitations

1. **Not tested on Windows or macOS.** The code uses `pathlib` and no shell-specific commands, so it should work, but this is not verified.
2. **Ollama collector not tested against a real Ollama server.** To verify:
   ```bash
   ollama pull llama3.2:3b
   python -m llm_failure_mining.collect_ollama --model llama3.2:3b --trials 1 --max-prompts 3
   ```
   Expected: "Saved 3 responses to data/collected/ollama_llama3.2_3b.json (accuracy …)". Then open the file and check that `parsed_answer` matches the number after `### Answer:` in each `model_response`.
3. **Analysing a small collected file:** the clustering stage needs at least `k_max × 5` = 50 long failed responses. For a small Ollama run, lower `clustering.k_max` and `clustering.min_words` in `config.toml`, or collect more trials.
4. **Clustering quality is low** (silhouette 0.162) and should be treated as exploratory.
5. **Style effects are correlations,** not causal effects, because styles were not run evenly across problems and models.
