# Verification
 
All checks below were actually run. The pipeline was developed and first tested on
4 November 2026 on Linux with Python 3.13, pandas 3.0, scikit-learn 1.9 and mlxtend 0.25.
On 17 November 2026 it was run again on Windows 11 with Python 3.12, with identical
results. It was not tested on macOS.
 
## Requirement checklist
 
| ID | Requirement | Type | Acceptance criterion | Evidence | Status |
|---|---|---|---|---|---|
| R1 | Reproducible data download | Essential | File matches pinned SHA-256 | `scripts/fetch_aiw_data.py` downloaded 45 MB and printed "Saved and verified" on Linux and on Windows; a second run printed "already present and verified" | Done |
| R2 | Validated loading | Essential | Invalid records dropped and counted | `tests/test_data.py` (7 tests); full data: 0 dropped of 49,838 | Done |
| R3 | Prompt, model and response features | Essential | Correct values for known cases | `tests/test_features.py` (25 tests incl. parametrised cases) | Done |
| R4 | Interpretable error taxonomy | Essential | Every failure gets one category | Shares in `results/summary.json` sum to 1 (checked in end-to-end test) | Done |
| R5 | Clustering of failed responses | Essential | k chosen by silhouette, summary consistent | `test_clustering_returns_consistent_summary`; full run: k = 10, silhouette 0.162 | Done |
| R6 | Association rules | Essential | Only target consequents, lift > 1 | `test_rules_have_single_target_consequent_and_lift_above_one`; full run: 25 rules | Done |
| R7 | Leakage-safe failure prediction | Essential | No response features; no shared groups | `test_prediction_features_exclude_response_information`, `test_grouped_split_never_shares_prompts` | Done |
| R8 | Baselines | Essential | Constant baseline = 0.5 ROC-AUC; priors reported | `test_evaluation_runs_and_constant_baseline_is_chance`; `results/prediction_summary.csv` | Done |
| R9 | Auto-generated report and figures | Essential | All output files created | `test_end_to_end_on_sample` checks 11 files | Done |
| R10 | Clear error messages | Essential | Missing data gives instructions and exit code 1 | `test_cli_reports_missing_data`; manual run shown below | Done |
| R11 | Optional Ollama collector | Optional | Correct request and output schema; works with a real server | `test_collect_with_mocked_ollama` (mocked); live run on Windows with llama3.2:3b: 315 responses (63 prompts × 5 trials) saved | Done |
| R12 | Documentation | Essential | README, explanation, verification, notices | Files in project root | Done |
| R13 | Compare a new model with the AIW reference | Optional | Error profiles sum to 1; report written | `tests/test_compare.py` (2 tests); live run on the llama3.2:3b data | Done |
 
## Commands run and results
 
On Linux, Python 3.13:
 
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
 
On Windows 11, Python 3.12:
 
```text
> python scripts/fetch_aiw_data.py
Saved and verified: ...\data\raw\AIW_AIW_plus.json
 
> python -m llm_failure_mining.cli run
    49838 responses, 75 models, 63 prompts
    best k = 10
    25 rules kept
Done in 37.2 s.
 
> python -m llm_failure_mining.collect_ollama --model llama3.2:3b --trials 5
Collecting 5 trial(s) for 63 prompts from llama3.2:3b
  315/315
Saved 315 responses to ...\data\collected\ollama_llama3.2_3b.json (accuracy 2.5%)
 
> python -m pytest -q
48 passed in 12.66s
 
> python -m llm_failure_mining.compare --new data\collected\ollama_llama3.2_3b.json
Report: ...\results\compare_ollama_llama3.2_3b\COMPARISON.md
```
 
## Mocked versus real tests
 
- **Real data:** loading, features, clustering, rules and prediction tests use the committed sample of 435 real AIW records. The full pipeline was run on all 49,838 records, on Linux and on Windows.
- **Mocked and live:** the Ollama collector test replaces the HTTP call with a fake response, which checks request content, answer parsing and output format. In addition, the collector was run against a real local Ollama server with llama3.2:3b (315 responses).
- **Synthetic:** `tests/test_compare.py` uses a small hand-made set of answers with known error types, so the expected shares can be checked exactly.
## Important results
 
### AIW data (75 models)
 
| Result | Value | Source |
|---|---|---|
| Overall accuracy | 7.2% | `results/summary.json` |
| Failures that are right − 1 | 58.0% (68.8% on sibling problems) | `results/summary.json` |
| Accuracy vs off-by-one share, 49 models | Spearman ρ = −0.058 | `results/summary.json` |
| Best cluster count | k = 10, silhouette 0.162 | `results/REPORT.md` |
| Unseen prompts, prior_by_model | ROC-AUC 0.837 ± 0.055, PR-AUC 0.314 ± 0.059 | `results/prediction_summary.csv` |
| Unseen prompts, gradient boosting with model id | ROC-AUC 0.845 ± 0.028, PR-AUC 0.453 ± 0.090 | `results/prediction_summary.csv` |
| Paired comparison vs prior | ROC-AUC better in 3/5 folds (+0.008), PR-AUC better in 4/5 folds (+0.140) | `results/REPORT.md` |
| Unseen models, best method | prior_by_family ROC-AUC 0.676 ± 0.093 | `results/prediction_summary.csv` |
 
### Extension: Llama 3.2 3B collected with Ollama
 
| Result | Llama 3.2 3B | AIW, all models | AIW, models < 10B | Source |
|---|---|---|---|---|
| Responses | 315 | 49,838 | 10,840 | `results/compare_ollama_llama3.2_3b/summary.csv` |
| Accuracy | 2.5% | 7.2% | 4.3% | same |
| Accuracy on sibling problems | 0% | 8.3% | 5.3% | same |
| Off-by-one share of sibling failures | 19.3% | 68.8% | 33.9% | same |
| No parsable answer | 27.6% | 5.5% | 8.0% | same |
 
## Unresolved problems and limitations
 
1. **Not tested on macOS.**
2. **Labels for collected data are automatic.** A response counts as correct only if the parsed number equals the right answer. The AIW labels also include the authors' manual checks, so collected and AIW data are not labelled in exactly the same way.
3. **Parser coverage for the Llama 3.2 3B answers is not yet checked by hand.** 27.6% of its answers had no parsable `### Answer:` number. To verify, print a few of them:
```bash
python -c "import json; d=json.load(open('data/collected/ollama_llama3.2_3b.json')); [print(r['model_response'][-300:], '\n-----') for r in [x for x in d if x['parsed_answer'] is None][:5]]"
```
   If the answers contain a number in another format, the parser should be extended and the data collected again.
4. **The Llama 3.2 3B result is one model and 315 answers.** It is an observation, not a general result. Its 2.5% correct answers all came from the cousin problems, which with so few correct answers is likely chance.
5. **Analysing a small collected file with the full pipeline does not work:** prediction needs at least two models and clustering needs at least `k_max × 5` = 50 long failed responses. Use `python -m llm_failure_mining.compare` for a single model.
6. **Clustering quality is low** (silhouette 0.162) and should be treated as exploratory.
7. **Style effects are correlations,** not causal effects, because styles were not run evenly across problems and models.
 