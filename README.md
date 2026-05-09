# Permission-to-Policy Checker

**Final project for CSC 7644: Applied LLM Development**
**Author:** Abdur Rahman Onik · Louisiana State University · `aonik1@lsu.edu`

LLM-based mismatch detection between Android app permissions and privacy
policy disclosures. Given a Google Play Store app ID, the system fetches
the app's requested permissions and its privacy policy, and uses a large
language model to decide — for each permission — whether the policy
adequately discloses the corresponding data collection. Verdicts are
grounded in verbatim quotes from the policy text to control hallucination.

The project also implements an **agentic** version of the same task: the
LLM is given six tools and drives the audit autonomously. The repository
provides a controlled empirical comparison of one-shot prompting vs.
agentic tool use on this task. See [`REPORT.md`](REPORT.md) for the full
technical report.

---

## Key Features

- **End-to-end audit pipeline** — Play Store metadata scrape, privacy-policy
  HTML fetching and cleanup, LLM classification, and structured per-permission
  reports.
- **Three-class verdict schema** — `covered` / `unclear` / `mismatch`,
  capturing not just the presence/absence of disclosure but also vague
  catch-all language (the discriminative class for compliance auditing).
- **Verbatim-quote hallucination control** — every covered/unclear verdict
  must cite a substring that exists literally in the policy text. Enforced
  inline (agent mode) and verified post-hoc (`eval/verify_quotes.py`).
- **Two operational modes** — single-prompt classification and a multi-step
  agent loop with six tools (`fetch_app_metadata`, `fetch_policy`,
  `search_policy`, `read_policy_section`, `verify_quote`, `submit_verdict`).
- **Seven interchangeable backends** — Anthropic Claude (one-shot + agent),
  Groq Llama 3.3 70B (one-shot + agent), local Ollama Llama 3.1 8B
  (one-shot + agent), keyword heuristic baseline, fine-tuned LoRA adapter,
  cached Claude predictions for deterministic eval.
- **Hand-labeled evaluation set** — 10 apps across 5 categories, 51 (app,
  permission) verdict instances. Precision / recall / F1 harness with
  per-class metrics and confusion matrices.
- **Cross-backend comparison** — side-by-side scripts (`compare_backends.py`,
  `ablation.py`) for one-shot vs. agent and base vs. fine-tuned analyses.
- **Streamlit UI and CLI** — interactive auditing of any Play Store app, plus
  scriptable runs for batch evaluation.
- **Fine-tuning track** — LoRA training pipeline (`finetune/`) with data
  preparation, training, and inference scripts, plus a base-vs-tuned ablation
  framework.

---

## Tech Stack and Architecture

### LLMs and APIs

- **Anthropic Claude Sonnet 4.5** via `anthropic` Python SDK (one-shot and
  native tool-use agent).
- **Llama 3.3 70B** via Groq's free-tier OpenAI-compatible API (one-shot
  and structured-tool-use agent).
- **Llama 3.1 8B** running locally via [Ollama](https://ollama.com) (one-shot
  and a custom JSON tool-calling protocol for the agent variant).

### Frameworks and Libraries

| Layer | Tool |
|---|---|
| App scraping | `google-play-scraper` |
| Policy HTML cleanup | `requests`, `beautifulsoup4` |
| Schema validation | `pydantic` (Verdict / PermissionVerdict / AppReport) |
| LLM clients | `anthropic`, `requests` (Groq + Ollama HTTP) |
| Web UI | `streamlit` |
| CLI | `argparse` (standard library) |
| Fine-tuning (optional) | `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes` |

### High-level architecture

```
Play Store ──▶ google-play-scraper ──▶ AppMetadata
                                          │
Privacy URL ─▶ requests + BeautifulSoup ─▶ policy_text
                                          │
                                  ┌───────┴───────┐
                                  ▼               ▼
                              Mode A          Mode B
                            (one-shot)       (agent)
                                  │               │
                                  │      6 tools (fetch / search /
                                  │      read / verify / submit)
                                  │               │
                                  └───────┬───────┘
                                          ▼
                                  per-permission JSON
                                  {covered | unclear | mismatch}
                                          │
                                          ▼
                            CLI / Streamlit UI / Eval harness
```

The two operational modes share the rubric, the permission glossary, the
output schema, and the evaluation harness — only the orchestration of the
LLM call differs. This makes Mode A vs. Mode B a clean comparison.

---

## Setup Instructions

### Prerequisites

- **Python 3.10 or newer** (developed and tested on 3.12).
- **pip** (for dependency installation).
- **Operating system:** any. Developed on Windows 11; the code is
  cross-platform Python with no OS-specific dependencies.
- **(Optional) [Ollama](https://ollama.com/download)** — only needed for the
  `llama` and `agent-llama` local backends.
- **(Optional) NVIDIA GPU with CUDA** — only needed for the `finetuned`
  fine-tuning track.

### 1. Clone and install

```bash
git clone <this-repository-url>
cd permission-policy-checker
pip install -r requirements.txt
```

For the optional fine-tuning track:

```bash
pip install -r finetune/requirements.txt
```

### 2. Configure API keys

The application reads API keys from environment variables. **Do not commit
keys to the repository.** Copy the example file and edit it:

```bash
# Linux / macOS
cp .env.example .env

# Windows PowerShell
copy .env.example .env
```

Fill in the variables you intend to use. **None of the keys are required —
the project runs on the local `heuristic` backend without any keys, and on
the local `llama` backend if Ollama is installed.**

| Environment variable | Purpose | Where to get one |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude Sonnet (`claude`, `agent-claude` backends) | <https://console.anthropic.com/> (paid) |
| `GROQ_API_KEY`      | Groq Llama 3.3 70B (`groq`, `agent-groq` backends) | <https://console.groq.com/keys> (free tier, no card required) |

Then load `.env` into your shell. The supplied scripts read the variables
directly via `os.environ`; you can either `export` them manually or use any
`.env` loader of your choice (e.g. `python-dotenv`, `direnv`).

### 3. (Optional) Install local Llama via Ollama

Required only for the `llama` and `agent-llama` backends:

```bash
# 1. Install Ollama from https://ollama.com/download
# 2. Pull the model (~4.9 GB, one-time)
ollama pull llama3.1:8b
# 3. Start the local server (it usually starts at install time on Windows)
ollama serve
```

---

## Running the Application

### Streamlit web UI

```bash
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Enter a Play Store app ID
(for example, `me.lyft.android`), select a backend from the sidebar
dropdown, and click **Audit App**.

### Command-line interface

```bash
# Local-only (no API keys, no LLM)
python check.py --app-id me.lyft.android --backend heuristic

# Free cloud LLM (set GROQ_API_KEY first)
python check.py --app-id me.lyft.android --backend groq
python check.py --app-id me.lyft.android --backend agent-groq

# Local Llama via Ollama
python check.py --app-id me.lyft.android --backend llama

# Anthropic Claude (set ANTHROPIC_API_KEY first)
python check.py --app-id me.lyft.android --backend claude
python check.py --app-id me.lyft.android --backend agent-claude

# Write the JSON report to a file instead of stdout
python check.py --app-id me.lyft.android --backend groq --out lyft.json
```

### Evaluation harness

```bash
# Run the full eval set with a chosen backend
python -m eval.run_eval --backend heuristic
python -m eval.run_eval --backend groq
python -m eval.run_eval --backend agent-groq
python -m eval.run_eval --backend claude-cached

# Cross-backend comparison reports
python -m eval.compare_backends     # heuristic vs. cached Claude
python -m eval.ablation             # base LLM vs. fine-tuned LoRA
python -m eval.verify_quotes        # verbatim-quote hallucination check
```

Eval outputs are written to `eval/results/`.

### Available `--backend` values

| Backend | Mode | Required key / dependency |
|---|---|---|
| `heuristic` | rules baseline | none |
| `claude` | one-shot | `ANTHROPIC_API_KEY` |
| `groq` | one-shot | `GROQ_API_KEY` |
| `llama` | one-shot | Ollama + `llama3.1:8b` |
| `claude-cached` | replay saved Claude predictions | none (file in repo) |
| `finetuned` | local LoRA adapter | `finetune/checkpoints/final/` after training |
| `local-hf` | untrained HF base model (for ablation) | `finetune/requirements.txt` |
| `agent-claude` | agent (Claude tool use) | `ANTHROPIC_API_KEY` |
| `agent-groq` | agent (OpenAI-style tool calls) | `GROQ_API_KEY` |
| `agent-llama` | agent (manual JSON protocol) | Ollama + `llama3.1:8b` |
| `auto` | picks the best available | varies |

---

## Repository Organization

```
permission-policy-checker/
├── app.py                      # Streamlit UI entry point
├── check.py                    # CLI entry point
├── requirements.txt            # Base Python dependencies
├── .env.example                # Template for environment variables
├── README.md                   # This file
├── REPORT.md                   # Full technical report
│
├── src/                        # Application logic
│   ├── schema.py               # Pydantic models (Verdict, PermissionVerdict, AppReport)
│   ├── prompts.py              # System prompt + permission glossary
│   ├── scraper.py              # google-play-scraper wrapper, metadata caching
│   ├── extractor.py            # HTTP fetch + BeautifulSoup policy text cleaner
│   ├── analyzer.py             # Backend dispatch (Claude / Groq / Llama / heuristic / etc.)
│   ├── agent_tools.py          # Six tools + AgentSession state for agent backends
│   └── agent.py                # Per-backend agent loops (Claude / Groq / Llama)
│
├── eval/                       # Evaluation harness and ground truth
│   ├── ground_truth.json       # 10 hand-labeled apps × 5 categories, 51 verdicts
│   ├── claude_predictions.json # Cached Claude verdicts per eval app
│   ├── run_eval.py             # Precision / recall / F1 + confusion matrix
│   ├── verify_quotes.py        # Verbatim-quote hallucination check
│   ├── compare_backends.py     # Side-by-side cross-backend comparison
│   ├── ablation.py             # Base LLM vs. fine-tuned LoRA ablation
│   └── results/                # Generated eval outputs (per backend)
│       ├── RESULTS.md          # Human-readable summary across all runs
│       ├── eval_*.json         # Per-backend metrics
│       ├── comparison.json
│       └── quote_verification.json
│
├── finetune/                   # Fine-tuning track (LoRA on Llama 3.x)
│   ├── prepare_data.py         # Ground truth → SFT chat-format JSONL
│   ├── train.py                # TRL SFTTrainer + PEFT LoRA training
│   ├── infer.py                # Adapter loader used by the finetuned backend
│   ├── requirements.txt        # Extra deps for fine-tuning
│   ├── README.md               # Fine-tuning runbook
│   ├── dataset/                # Generated train.jsonl + val.jsonl
│   └── checkpoints/            # Saved LoRA adapters (created by training)
│
├── data/
│   ├── samples/                # Sample app metadata + policy text
│   └── cache/                  # Local Play Store scrape cache (gitignored)
│
└── tests/                      # Test placeholder
```

**Where a grader should look first:**

- For **what the system does and how to run it** → this README.
- For the **technical report** (methodology, results, discussion) →
  [`REPORT.md`](REPORT.md).
- For the **core LLM logic** → `src/analyzer.py` (one-shot dispatch),
  `src/agent.py` and `src/agent_tools.py` (agent track).
- For the **prompt design** → `src/prompts.py`.
- For **evaluation results** →
  [`eval/results/RESULTS.md`](eval/results/RESULTS.md).

---

## Headline Results (10-app eval set, 51 verdicts)

| Backend                    | Mode      | Accuracy | Macro-F1 |
|----------------------------|-----------|---------:|---------:|
| Heuristic baseline         | rules     |  0.706   |  0.495   |
| Local Llama 3.1 8B         | one-shot  |  0.725   |  0.524   |
| Groq Llama 3.3 70B         | one-shot  |  0.686   |  0.559   |
| Groq Llama 3.3 70B         | **agent** |  0.431   |  0.243   |
| Claude Sonnet 4.5 (cached) | one-shot  |  1.000   |  1.000   |

Quote-grounding pass rate (Claude on eval set): **51 / 51 verbatim**.

The most counterintuitive finding: **agentic tool use underperforms one-shot
prompting** on this task using the same model (Llama 3.3 70B), driven by
keyword-search brittleness, verbatim-quote rigidity, and multi-step error
compounding. Full discussion in [`REPORT.md`](REPORT.md) §6 and
[`eval/results/RESULTS.md`](eval/results/RESULTS.md).

---

## Ethics

This is a **screening instrument, not a legal verdict**. Outputs flag
potential disclosure gaps for human review; they are not compliance
findings. We frame results at the category level rather than naming
individual developers, except where mismatches are egregious. The system
reads only public Play Store metadata and developer-published privacy
policies — no user data is collected, no apps are installed, and no app
backends are interacted with.

---

## Attributions and Citations

### External libraries

The project uses the following open-source libraries; full versions are
pinned in `requirements.txt` and `finetune/requirements.txt`:

- `google-play-scraper` (Play Store scraping)
- `beautifulsoup4` and `requests` (HTML fetching and cleanup)
- `pydantic` (output schema validation)
- `anthropic` (Claude API client)
- `streamlit` (web UI)
- `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `datasets`
  (fine-tuning track)

### Models

- **Claude Sonnet 4.5** — Anthropic.
- **Llama 3.1 8B Instruct, Llama 3.3 70B Instruct** — Meta AI. Used under
  Meta's Llama 3 community license. The local 8B model is served via
  [Ollama](https://ollama.com); the 70B model is served via
  [Groq](https://groq.com)'s free-tier inference.

### Research literature

The project's design and evaluation is informed by the following published
work, cited fully in [`REPORT.md`](REPORT.md):

- Felt et al. (2012) — Android permission comprehension survey.
- Slavin et al. (2016) — Privacy policy violation detection via Android code analysis.
- Yu et al. (2016) — PPChecker: incomplete and inconsistent privacy policies.
- Zimmeck et al. (2017, 2019) — Automated analysis of privacy requirements (MAPS).
- Tang et al. (2024) — LLMs for privacy policy analysis at scale.
- Touvron et al. (2023); Grattafiori et al. (2024) — LLaMA / Llama 3 model families.
- Yao et al. (2022) — ReAct agent framework.
- Schick et al. (2023) — Toolformer.
- Wei et al. (2022) — Chain-of-thought prompting.
- Lewis et al. (2020) — Retrieval-augmented generation.
- Bommasani et al. (2021) — Foundation Models risks/opportunities survey.

### Adapted code

The analyzer's tolerant JSON-extraction logic (`_parse_model_json` in
`src/analyzer.py`) is a custom implementation following common patterns
documented in the Anthropic and OpenAI cookbooks for parsing
JSON-mode outputs that occasionally include leading/trailing prose.

The Streamlit UI layout follows the patterns documented in the official
Streamlit gallery and tutorial pages.

No code was copied verbatim from external repositories; all source files
in `src/`, `eval/`, and `finetune/` are original to this project.

### AI assistance disclosure

This project was developed with assistance from AI coding tools (Claude
via the Claude Code CLI). AI assistance was used for code scaffolding,
debugging, methodology drafting, and documentation. Research design,
problem framing, ground-truth labeling, model-selection rationale,
experimental decisions, and analysis of results are the author's own.
All AI-generated code was reviewed, modified, and integrated by the
author. Experimental results were produced by running the code on the
author's hardware and accounts.

---

## License

This project is released for academic review as part of CSC 7644 final
submission. External libraries and models retain their original licenses.
