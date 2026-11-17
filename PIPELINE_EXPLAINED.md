# How the Pipeline Works, Explained Simply

This guide explains the whole project so you can present it and answer follow-up
questions. Technical terms are defined the first time they appear, and each
section ends with a short summary.

---

## 1. The one-minute explanation

Large language models (LLMs) can write essays and code, yet many of them fail a
riddle a child can solve: *"Alice has 4 sisters and 1 brother. How many sisters
does Alice's brother have?"* The answer is 5, because Alice is also his sister.

The AIW research team asked 75 models these questions tens of thousands of
times and published every answer. This project treats those 49,838 answers as a
dataset and uses data mining to find out **how** the models fail, **when** they
fail, and whether we can **predict** failure in advance.

The headline: in 58% of all failures, and 69% on the sibling problems, the model
answers exactly one less than the truth. It forgets to count Alice.

**Summary:** we don't train an LLM. We analyse how existing LLMs fail, using
classic data mining methods on published, labelled answers.

---

## 2. The problem and why this approach makes sense

Benchmarks usually report a single number, accuracy. Accuracy tells you *that*
a model is wrong, not *why*. For trustworthy AI we need to know whether
failures are random noise or systematic. Systematic failures can be detected,
explained and perhaps fixed.

Data mining fits because:

- the data already exists, with labels from the AIW authors;
- the answers are numbers, so many errors can be explained exactly with simple rules;
- we have rich context for every answer (prompt wording, model family, size), which suits rule mining and classification.

**Summary:** accuracy alone hides the structure of errors; data mining reveals it.

---

## 3. An everyday analogy

Think of a driving school with 75 learner drivers and one tricky roundabout.

- **Accuracy** is "how many learners got through the roundabout correctly."
- **Error typing** is the examiner's note on *what* went wrong: wrong exit, forgot to indicate, stopped in the middle.
- **Clustering** is reading all the learners' explanations and grouping similar excuses.
- **Association rules** are patterns like "learners from School X, in the rain, usually take the wrong exit."
- **Prediction** asks: from the weather and the learner's school, can we guess before the drive who will fail?

**Where the analogy breaks:** learner drivers improve with practice, but these
models are frozen; each answer is independent. And unlike drivers, a model can
give a different answer to the identical question on the next try.

**Summary:** we look at the same failures from four angles: what, how, when, and can we predict it.

---

## 4. Pipeline diagram

```
data/raw/AIW_AIW_plus.json  (49,838 records)
            │
            ▼
   data.py  ── load, validate, normalise types, add prompt_key
            │
            ▼
 features.py ── prompt features ─┐
            ├── model features ──┤  one wide table
            └── response features + error_type
            │
   ┌────────┼──────────────┬─────────────────┐
   ▼        ▼              ▼                 ▼
report.py clustering.py  rules.py         predict.py
statistics TF-IDF→SVD→    Apriori on       grouped CV:
& figures  k-means on     one-hot          unseen prompts /
           failures       "baskets"        unseen models
   │        │              │                 │
   └────────┴──────┬───────┴─────────────────┘
                   ▼
          results/REPORT.md + CSV + PNG
```

`cli.py` runs these stages in order (numbered 1/6 to 6/6 in the console).

**Summary:** load, describe each answer, then analyse it four ways, then write one report.

---

## 5. What each file does

| File | Purpose |
|---|---|
| `scripts/fetch_aiw_data.py` | Downloads the dataset at a fixed commit and checks its SHA-256 fingerprint |
| `config.toml` | All settings you may want to change |
| `llm_failure_mining/config.py` | Reads and validates `config.toml` into a `Config` object |
| `llm_failure_mining/data.py` | Loads records, drops invalid ones, normalises types |
| `llm_failure_mining/features.py` | Adds prompt, model and response features, including `error_type` |
| `llm_failure_mining/clustering.py` | Groups failed responses by their wording |
| `llm_failure_mining/rules.py` | Finds association rules with Apriori |
| `llm_failure_mining/predict.py` | Trains and evaluates failure predictors without leakage |
| `llm_failure_mining/report.py` | Statistics, charts and `REPORT.md` |
| `llm_failure_mining/cli.py` | Command-line entry point that runs everything |
| `llm_failure_mining/collect_ollama.py` | Optional: get new answers from your own local models |
| `tests/` | Automated tests |

---

## 6. Stage by stage

### 6.1 Loading and validation (`data.py`)

- **Input:** a JSON list of records with `model`, `prompt`, `model_response`, `parsed_answer`, `right_answer`, `correct`.
- **Processing:** `to_dataframe` skips records with missing fields or a non-numeric right answer and counts them in `df.attrs["dropped_records"]` (0 for the real data). It converts strings like `"2.0"` to numbers and `"True"` to booleans. It gives every distinct prompt text an integer `prompt_key`.
- **Output:** a pandas DataFrame, one row per answer.

**Why no de-duplication?** The AIW authors ran each prompt many times per model.
A short answer such as `### Answer: 4` naturally repeats, and 16,402 rows share
an identical (model, prompt, response) triple. Those are real repeated trials, not
errors, so we keep them. We handle the risk they create (leakage) in the prediction stage.

**Label note:** `correct` is the authors' label. In 274 cases the parsed number
is right but `correct` is False; the authors checked these by hand and found the
reasoning was wrong. We keep their judgment and call these `right_number_wrong_reasoning`.

**Summary:** clean, typed table; repeated trials kept on purpose.

### 6.2 Feature engineering (`features.py`)

A **feature** is a column that describes a row in a way a method can use.

**Prompt features** (`prompt_features`):

- `problem_type`: `aiw_siblings` or `aiw_plus_cousins` (harder, contains "cousin").
- `prompt_style`: how the instruction is phrased, found by `instruction_style` using marker phrases in `STYLE_RULES`, checked in order (first match wins): `base_completion`, `persona_scientist`, `gender_clarified`, `confidence_request`, `think_step_by_step`, `restricted_format`, `standard`, else `bare_question`.
- `mention_order`: whether sisters or brothers are mentioned first.
- `numbers_as_words`: "four" instead of "4".
- `subject_name`: Alice or Haret (a name the authors used to test name sensitivity).
- `n_stated_numbers`, `prompt_words`.

**Model features** (`model_features`):

- `family`: from name patterns in `FAMILY_RULES` (e.g. `open-mixtral-8x7b` → `mistral`).
- `size_b`: parameters in billions from the name; mixture-of-experts names like `8x22b` count as 8 × 22 = 176. Closed models such as GPT-4 have no public size → NaN → `size_bucket = unknown_closed`.
- `is_base_model`, `is_code_model`.

**Response features:** `response_words`, `length_bucket`, `has_answer_marker`, and the most important one, `error_type`.

**The error taxonomy** (`error_type`) checks these explanations in order:

| Error type | Rule | Meaning in the sibling problem |
|---|---|---|
| `correct` | authors' label is True | |
| `no_parsable_answer` | no number could be parsed | refused, rambled, wrong format |
| `right_number_wrong_reasoning` | number right, label False | lucky guess |
| `off_by_one_low` | answer = right − 1 | **forgot to count Alice** |
| `off_by_one_high` | answer = right + 1 | counted someone twice |
| `copies_given_number` | answer equals a number in the problem | e.g. answered with the number of brothers |
| `adds_given_numbers` | answer = sum of two stated numbers | added sisters and brothers |
| `other_wrong` | anything else | |

Order matters: in "4 sisters, 1 brother", the answer 4 is both "right − 1" and
"a given number". We check `off_by_one_low` first because it is the more specific explanation.

**Summary:** every answer gets a description of the question, the model, and the kind of mistake.

### 6.3 Descriptive statistics (`report.py`)

`describe` computes overall accuracy, accuracy by problem type and style, error
shares, and the best and worst models (only models with at least 200 answers).

It also tests a **hypothesis** in `_accuracy_vs_off_by_one`: *do stronger models fail
more consistently with the off-by-one error?* For each of the 49 models with at
least 200 failures, it compares accuracy with the share of off-by-one failures using
**Spearman rank correlation** (ρ, from −1 to +1; it compares rankings, so it is robust
to outliers). Result: ρ = −0.058, so no evidence for the idea.

**Summary:** the numbers that set the scene, plus one honest hypothesis test.

### 6.4 Clustering failed responses (`clustering.py`)

**Goal:** find groups of failed answers that use similar language, without labels.
This is *unsupervised* learning.

1. `select_failures`: failed answers with at least 20 words (short "### Answer: 4" replies have no reasoning text). Up to 8,000 sampled with a fixed seed.
2. **TF-IDF** (term frequency–inverse document frequency): turns each text into a vector where a word counts more if it is frequent in that text but rare overall. We use words and two-word phrases, ignore numbers (they change between prompts) and English stop words.
3. **Truncated SVD**, also called latent semantic analysis: compresses thousands of word columns into 100 "topic" dimensions, so similar wordings end up close together.
4. **k-means**: splits the vectors into k groups around k centres.
5. **Choosing k**: we try k = 4 to 10 and keep the highest **silhouette score** (how much closer each point is to its own cluster than to the next one; from −1 to 1).
6. Each cluster is described by its top TF-IDF terms, its dominant `error_type`, its top models, and the example closest to its centre.

**Result:** k = 10 had the highest silhouette, 0.162. That is low, and k sat at the
top of the search range, so the clusters overlap and partly mirror the problem wording
(for example a "Haret" cluster and a "cousins" cluster). Still, two useful groups appear:
answers that claim the problem lacks information (`no_parsable_answer`), and apologetic
or off-topic answers (top models in that cluster: phi-2, codellama-70b-instruct, gemma-7b-it).

**Why TF-IDF instead of neural embeddings?** No model download, identical results on
every machine, and the top terms are directly readable. Sentence embeddings would
probably separate reasoning patterns better; that is listed as an improvement.

**Summary:** we grouped failure texts; the grouping is weak but shows two distinct non-arithmetic failure modes.

### 6.5 Association rules (`rules.py`)

**Goal:** find combinations of conditions that go with an outcome.

- `build_transactions` turns each answer into a "basket" of true/false items, e.g. `{problem=aiw_siblings, style=standard, family=claude, size=unknown_closed, length=medium_20-150w, outcome=FAIL, error=off_by_one_low}`.
- **Apriori** finds item sets present in at least 2% of baskets (`min_support`).
- From those, rules `A → B` are formed. We keep only rules whose right side is a single `outcome=` or `error=` item and whose left side contains none, with confidence ≥ 0.5 and **lift > 1**.

Definitions:
- **support** = share of all answers containing both sides;
- **confidence** = P(B | A);
- **lift** = confidence ÷ P(B). Lift 1.8 means "1.8 times more often than usual".

Because 93% of answers fail, a rule "→ FAIL" almost always has high confidence. That's why we sort by **lift**, not confidence.

**Strongest rules (full data):**
- Short Qwen answers → `other_wrong`, lift 3.54. When Qwen models answer briefly, they often give a number that fits none of our error explanations.
- Mistral family, sibling problem, more than 100B parameters → `off_by_one_low`, confidence 0.97, lift 1.81.

**Summary:** rules show which model families make which mistake under which conditions.

### 6.6 Predicting failure (`predict.py`)

**Goal:** from information known *before* answering (prompt and model features), predict whether the answer will be correct. This is *supervised* learning.

`prediction_frame` builds the inputs and deliberately excludes the response,
the parsed answer and the error type, which would reveal the label.

**Data leakage** means the test data secretly contains information from training.
Here, the same prompt was asked many times. A random split would put copies of one
question in both training and test sets. We prevent this with **GroupKFold**:

- `unseen_prompts`: grouped by `prompt_key`, so test prompts never appear in training.
- `unseen_models`: grouped by `model`, so test models never appear in training.

Methods compared in each fold:

| Method | Idea |
|---|---|
| `constant_baseline` | predict the training base rate for everyone (ROC-AUC 0.5 by definition) |
| `prior_by_model` / `prior_by_family` | predict the training accuracy of the same model (or family) |
| `logistic_regression` | linear model on one-hot categories and scaled numbers |
| `gradient_boosting` | `HistGradientBoostingClassifier`, many small decision trees |
| `gradient_boosting_with_model_id` | same, plus the exact model name (only in `unseen_prompts`, where it is fair) |

`class_weight="balanced"` makes the rare "correct" class count as much as "fail" during training.

Metrics:
- **ROC-AUC**: the chance that a random correct answer gets a higher score than a random failure (0.5 = guessing, 1 = perfect).
- **PR-AUC** (average precision): focuses on finding the rare positives; the guessing level equals the positive rate (about 0.07).
- **Balanced accuracy**: the average of the hit rates on both classes.

**Results:**

| Setting | Method | ROC-AUC | PR-AUC |
|---|---|---|---|
| unseen prompts | prior_by_model | 0.837 ± 0.055 | 0.314 ± 0.059 |
| unseen prompts | gradient_boosting_with_model_id | 0.845 ± 0.028 | 0.453 ± 0.090 |
| unseen models | prior_by_family | 0.676 ± 0.093 | 0.126 ± 0.078 |
| unseen models | logistic_regression | 0.675 ± 0.050 | 0.116 ± 0.056 |

Interpretation: for new prompts, *which model* explains most of the signal. Prompt
features add a little ROC-AUC (within the fold variation) but clearly improve PR-AUC
(+0.140, better in 4 of 5 folds), so they help pick out the rare correct answers. For
new models, nothing beats knowing the family.

**Permutation importance** (`feature_importance`) shuffles one input column at a time on a
held-out fold and measures the drop in ROC-AUC. `family` matters most (0.232), followed by
`mention_order` (0.070) and `problem_type` (0.050). `mention_order` is probably partly a
stand-in for specific prompts, because each wording was tested with specific numbers.

**Summary:** failure is predictable mainly from the model; the prompt adds useful but modest extra signal.

---

## 7. One example through every stage

**Raw record** (Claude 3 Opus):

- Prompt: *"Alice has 4 sisters and she also has 3 brothers. How many sisters does Alice's brother have? Solve this problem and provide the final answer in following form: "### Answer: "."*
- Response (shortened): *"…there are a total of 8 siblings (including Alice). If we consider any one of Alice's brothers, he would have all the sisters that Alice has, which is 4. … ### Answer: 4"*
- `parsed_answer` 4.0, `right_answer` 5.0, `correct` False

**After `features.py`:**

```python
{'problem_type': 'aiw_siblings', 'prompt_style': 'standard', 'mention_order': 'sisters_first',
 'subject_name': 'Alice', 'numbers_as_words': False, 'n_stated_numbers': 2, 'prompt_words': 31,
 'family': 'claude', 'size_b': nan, 'size_bucket': 'unknown_closed',
 'error_type': 'off_by_one_low', 'response_words': 86, 'length_bucket': 'medium_20-150w'}
```

`stated_numbers` found `[4, 3]`. The answer 4 equals right − 1, so `off_by_one_low`.
Notice the model even says "including Alice" and then forgets her.

**As an Apriori basket:**
`['problem=aiw_siblings', 'style=standard', 'family=claude', 'size=unknown_closed', 'length=medium_20-150w', 'outcome=FAIL', 'error=off_by_one_low']`

**As prediction input** (no response information):
`{'prompt_style': 'standard', 'problem_type': 'aiw_siblings', 'family': 'claude', 'size_bucket': 'unknown_closed', 'size_known': 0.0, 'log_size_b': 0.0, ...}`

**Summary:** one answer becomes a table row, a basket, and a prediction input, each used by a different method.

---

## 8. How errors are handled

| Situation | Handling |
|---|---|
| Data file missing | `FileNotFoundError` with instructions; CLI exits with code 1 |
| Download corrupted | `fetch_aiw_data.py` checks SHA-256 and deletes the partial file |
| Invalid record | skipped and counted in `dropped_records` |
| Invalid config value | `load_config` raises `ValueError` naming the key |
| Too few failures to cluster | `ValueError` advising to lower `min_words` |
| A fold has only one class | metrics are `NaN` for that fold instead of crashing |
| Ollama not reachable | up to 3 attempts with growing waits, then a clear error message |

---

## 9. How configuration changes behaviour

| Setting | Effect of increasing it |
|---|---|
| `clustering.min_words` | fewer, longer responses are clustered |
| `clustering.k_max` | more cluster counts are tried |
| `rules.min_support` | fewer but more common rules |
| `rules.min_confidence` | only stronger rules survive |
| `prediction.n_splits` | more folds, smaller test sets, better variance estimate |
| `run.seed` | different sampling and model initialisation; results should change only slightly |

---

## 10. Why these libraries and choices

- **pandas** for tables, **scikit-learn** for TF-IDF, SVD, k-means, classifiers and cross-validation, **mlxtend** for Apriori, **matplotlib** for charts. All are standard, well tested and free.
- **Rule-based error types** before machine learning: numeric answers allow exact, explainable categories. This matters for trustworthy AI.
- **Grouped cross-validation** instead of a random split: avoids leakage from repeated trials.
- **Priors as baselines:** a learned model is only useful if it beats "how good is this model in general".

---

## 11. Limitations and possible improvements

**Limitations**
- Two problem families, 63 prompts: narrow.
- Style and content are confounded (not all combinations were run).
- Clusters are weak (silhouette 0.16).
- Error types are based on the final number, not the reasoning steps.
- Closed-model sizes are unknown.

**Improvements**
- Sentence-transformer embeddings for clustering.
- Annotate a few hundred reasoning chains by hand to validate the error taxonomy.
- Collect new answers from current open models with `collect_ollama.py` and test whether the off-by-one pattern still holds.
- A controlled experiment: every style crossed with every problem, to remove the confound.

---

## 12. Glossary

- **LLM**: large language model, a neural network trained to predict text.
- **Feature**: a column describing a row.
- **One-hot encoding**: turning a category into several 0/1 columns.
- **TF-IDF**: a word weighting that favours words frequent in one text and rare overall.
- **SVD / LSA**: compressing many word columns into fewer topic dimensions.
- **k-means**: grouping points around k centres.
- **Silhouette score**: how well separated clusters are (−1 to 1).
- **Apriori**: an algorithm that finds frequent item combinations.
- **Support, confidence, lift**: how common a rule is, how reliable, and how much above chance.
- **Supervised / unsupervised**: learning with or without labels.
- **Data leakage**: test information sneaking into training.
- **GroupKFold**: cross-validation that keeps whole groups together.
- **ROC-AUC, PR-AUC, balanced accuracy**: classification metrics (section 6.6).
- **Permutation importance**: how much a score drops when one feature is shuffled.
- **Spearman correlation**: correlation of rankings.

---

## 13. Explain it in your own words

1. **Why don't we remove duplicate rows?**
   *Suggested answer:* They are real repeated trials of the same prompt. Removing them would distort accuracy. We handle the risk they create with grouped splits.

2. **Why is `off_by_one_low` checked before `copies_given_number`?**
   *Suggested answer:* In the sibling problem, right − 1 equals the stated number of sisters, so both rules match. "Forgot Alice" is the more specific, meaningful explanation.

3. **Why sort rules by lift and not confidence?**
   *Suggested answer:* 93% of answers fail, so almost any rule predicting failure has high confidence. Lift shows whether a rule beats that base rate.

4. **What would go wrong with a random train/test split?**
   *Suggested answer:* Copies of the same prompt would appear in both sets, so the model would be tested on questions it already saw, and the score would be too optimistic.

5. **What does PR-AUC tell us that ROC-AUC doesn't?**
   *Suggested answer:* How well we find the rare correct answers. With only 7% positives, ROC-AUC can look good even when precision is poor.

6. **Is the clustering a strong result? Why or why not?**
   *Suggested answer:* No, the silhouette is only 0.16 and the clusters partly mirror problem wording. The rule-based error types are the stronger result; clustering adds two qualitative failure modes.
