# Third-Party Notices

## AIW: "Alice in Wonderland" benchmark data (LAION-AI)

- Repository: https://github.com/LAION-AI/AIW
- Commit used: `4d5d079b773ae6ab7d2eb852971ea3fda0ea1ce8` (4 February 2026)
- License: Apache License 2.0
- Paper: Nezhurina, Cipolina-Kun, Cherti, Jitsev. *Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models.* arXiv:2406.02061.

**What we reuse**
- The data file `collected_responses/AIW_AIW_plus.json` (49,838 model responses with the authors' parsed answers and correctness labels). It is downloaded by `scripts/fetch_aiw_data.py` and verified by SHA-256 checksum `e84ce307d14657cb4650ae6f4cbfab995e7745e5ce5721b0cd6252080f3c2cc1`.
- A small subset of that file is committed as `data/sample/aiw_sample.json` (435 records, selected by `python -m llm_failure_mining.cli make-sample`). The records are unmodified.
- The `### Answer:` output convention used in the AIW prompts, which our optional Ollama collector also parses.

**What we did not reuse**
No source code from the AIW repository is copied into this project. All code under `llm_failure_mining/`, `scripts/` and `tests/` was written for this project.

**Modifications**
None to the data itself. Our code reads the records, normalises field types (for example `"2.0"` to `2.0`), and adds derived columns such as instruction style, model family and error type.

Apache License 2.0 text: https://www.apache.org/licenses/LICENSE-2.0

## Python libraries

Installed from PyPI through `requirements.txt` and used as normal dependencies, not copied into this repository: pandas, NumPy, scikit-learn and mlxtend (BSD-3-Clause), matplotlib (PSF-based Matplotlib license) and pytest (MIT). Each library keeps its own license.
