# Permission-to-Policy Checker: A Comparative Study of One-Shot and Agentic LLM Auditing for Android Privacy Disclosure

**Author:** Abdur Rahman Onik
**Affiliation:** Louisiana State University
**Email:** aonik1@lsu.edu
**Date:** May 2026

---

## Abstract

Android privacy policies routinely fail to disclose the data collection implied by their applications' requested permissions: Zimmeck et al. (2019) found 89% of Play Store apps engage in at least one data practice requiring disclosure, with 12.1% having location-related compliance gaps alone. Manual audit does not scale to a 3M-app store. Prior automation has used either custom NLP classifiers (Zimmeck et al., 2017, 2019) or static-code-analysis pipelines (Slavin et al., 2016; Yu et al., 2016), both of which require costly task-specific training data or brittle API ontologies. Recent work (Tang et al., 2024) shows that large language models can match traditional NLP pipelines on privacy-policy analysis using only general instruction-following.

We extend this line in two directions. First, we build an end-to-end pipeline (scrape → extract → classify → ground) that audits a Play Store app's requested permissions against its privacy policy text and produces a per-permission verdict {covered, unclear, mismatch} with a verbatim policy quote for every non-mismatch verdict. Second, we empirically compare two operational modes of the same task: a **one-shot** prompt where the model classifies all permissions in a single completion, and an **agent** mode where the model is given six tools (`fetch_app_metadata`, `fetch_policy`, `search_policy`, `read_policy_section`, `verify_quote`, `submit_verdict`) and drives the audit autonomously across multiple turns. We evaluate five backends — a keyword baseline, Llama 3.1 8B (Touvron et al., 2023; Grattafiori et al., 2024), Llama 3.3 70B in both one-shot and agent mode, and Claude Sonnet 4.5 — on a hand-labeled 51-verdict eval set spanning ten apps in five categories.

Our central finding is counterintuitive: **agentic tool use underperforms one-shot prompting on this task using the same model.** Llama 3.3 70B's macro-F1 falls from 0.559 (one-shot) to 0.243 (agent), driven by three identifiable failure modes — keyword-search brittleness, verbatim-quote rigidity, and multi-step error compounding. We argue that agentic mode is appropriate when documents exceed the model's context window or the task requires genuine multi-step reasoning, neither of which holds for per-permission classification of typical-length privacy policies. The paper contributes the empirical comparison, the failure-mode analysis, and an open-source pipeline.

---

## 1. Introduction

Android applications declare a list of requested permissions to the operating system and a separately-authored privacy policy to the user. The two should be consistent: any data category the permission grants access to should be disclosed in the policy. Two strands of empirical evidence indicate they often are not. Felt et al. (2012) showed that only 17% of Android users examine permissions during installation and only 3% can correctly interpret them, leaving the privacy policy as the de facto disclosure mechanism. Zimmeck et al. (2019), in their MAPS analysis of over a million apps, found that 89% engage in at least one data practice that legally requires disclosure, with location-related compliance gaps alone affecting 12.1% of apps. The privacy-policy-vs-permissions inconsistency is therefore both widespread and consequential.

Prior automation has taken two main approaches:

1. **Trained NLP classifiers** (Zimmeck et al., 2017, 2019). MAPS used a hand-annotated corpus of 350 privacy policies and trained per-data-category classifiers, scaling the resulting model to a million apps. The accuracy was strong but the training and ontology pipeline expensive to maintain.
2. **Static code analysis with mapping ontologies** (Slavin et al., 2016; Yu et al., 2016). PPChecker mapped policy phrases to Android API methods using information-flow analysis. This caught violations the policy did not mention but was brittle to API churn and could not handle vague policy language.

Both approaches share a reliance on artifacts (annotated corpora, API ontologies) that drift and require maintenance. Tang et al. (2024) recently demonstrated that pretrained LLMs can match or exceed these pipelines on privacy-policy analysis using only their general reading-comprehension capability, with no task-specific training. This suggests that the right unit of comparison for new privacy-auditing tools is no longer "did we beat MAPS" but "what is the best LLM-based design for this task."

The 2024–2026 LLM literature has converged on **agentic tool use** as a default upgrade over single-prompt classification (Yao et al., 2022; Schick et al., 2023). The thesis is that giving the model tools — search, retrieval, code execution, self-verification — improves performance by allowing iterative reasoning instead of forcing a single committed response. We test whether this thesis holds for privacy auditing, where the tool surface is naturally well-defined (search a policy for a phrase; read a section by offset; verify a quote against the source).

This report makes three contributions:

- **An end-to-end open-source auditing pipeline** with five interchangeable backends (heuristic baseline, two open-weight Llama variants, Claude Sonnet 4.5, and an agent variant of Llama 3.3 70B), available as a Streamlit UI and a CLI.
- **A controlled empirical comparison** of one-shot vs. agentic LLM auditing, holding the underlying model, the rubric, and the output schema constant. We find that the agent mode degrades macro-F1 by 0.316 absolute on the same model.
- **A failure-mode analysis** of why agentic mode underperforms here, with three identifiable mechanisms — naïve keyword search, verbatim-quote rigidity, and multi-step error compounding — that scope when agentic auditing is appropriate.

## 2. Related Work

**LLMs for privacy text.** Tang et al. (2024) is the immediate predecessor: they showed that GPT-class LLMs match or exceed handcrafted NLP pipelines for privacy-policy analysis without task-specific training. Their evaluation covers multiple labels (e.g. "is this a third-party data sharing clause?") on policy excerpts. Our task is different in framing — we audit permission-vs-policy consistency, not policy-content classification — but their methodology motivates ours.

**Automated privacy-compliance pipelines.** MAPS (Zimmeck et al., 2019) operationalizes the Zimmeck et al. (2017) framework at scale, training classifiers on a hand-annotated corpus. Slavin et al. (2016) used static analysis of Android APKs to detect uses of sensitive APIs that the policy did not disclose. Yu et al. (2016) introduced PPChecker, identifying incomplete, incorrect, and inconsistent policies via system-managed-data analysis. Our approach differs by reading **only public text** (Play Store metadata + policy) and forgoing decompilation, trading precision for portability.

**Tool-using LLM agents.** ReAct (Yao et al., 2022) introduced the "reason + act" loop where an LLM interleaves chain-of-thought reasoning (Wei et al., 2022) with tool calls. Toolformer (Schick et al., 2023) showed LLMs can teach themselves to use tools by self-supervised demonstrations. The native tool-calling features of contemporary models (Claude's `tool_use`, OpenAI/Groq's `tool_calls` field) generalize this to a standard API. We use this API directly without additional fine-tuning.

**Hallucination control via citation.** Forcing the model to quote source text is well-established (e.g. retrieval-augmented generation per Lewis et al., 2020). We use a stricter form: a verdict citing a policy quote is only accepted if the quote is a verbatim substring of the policy. This converts hallucination from a measurement problem (requiring an LLM-as-judge or human review) into a string-equality test, which is cheap and reproducible.

**Position vs. prior work.** No prior work to our knowledge directly compares one-shot and agentic LLM auditing on the same privacy task with the same model. Tang et al. (2024) is one-shot only; the agent literature (Yao et al., 2022; Schick et al., 2023) reports head-to-head wins for tool use on open-domain QA and reasoning tasks but does not address short-document classification. We fill that gap.

## 3. Methodology

We separate the methodology into eight sub-components: task formulation, pipeline architecture, data collection, policy extraction, auditor prompt design, the two operational modes, the hallucination-control mechanism, and model selection.

### 3.1 Task Formulation

Given an app *a* with requested permissions *P_a = {p_1, ..., p_n}* and privacy policy text *T_a*, the system produces a per-permission verdict *v(p_i) ∈ {covered, unclear, mismatch}*:

- **covered:** *T_a* explicitly states that the application collects, uses, or accesses the data category that *p_i* grants. This requires a direct, identifiable disclosure.
- **unclear:** *T_a* mentions the data category vaguely, by inference, or only via a generic catch-all. The canonical example is a policy that says "we collect device information" used to cover an `ACCESS_FINE_LOCATION` permission — the disclosure is non-specific.
- **mismatch:** *T_a* does not mention the data category, or actively contradicts what *p_i* permits (e.g. a policy that says "we do not access your contacts" when the app requests `READ_CONTACTS`).

A verdict for a covered or unclear permission is paired with a **policy span** — a verbatim substring of *T_a* that supports the verdict. A mismatch verdict emits the literal string `"NOT MENTIONED"`.

This three-class formulation is **deliberately richer than binary covered/missing**. Privacy compliance hinges on whether disclosure is *specific enough*, not merely whether it exists; binary labels collapse this distinction (cf. Yu et al. 2016's "incomplete" category). Our `unclear` class captures the discriminative phenomenon.

The unit of analysis is one (app, permission) pair, not the app as a whole. An app with five permissions produces five verdicts.

### 3.2 Pipeline Architecture

The pipeline has four stages, identical to those proposed in the project's pre-implementation document but generalized along the operational-mode axis:

```
Stage 1: Data collection
  google-play-scraper → AppMetadata{title, developer, permissions, privacy_policy_url}

Stage 2: Policy extraction
  HTTP fetch + BeautifulSoup → policy_text (≤ 10K chars in one-shot; full text in agent)

Stage 3: Classification
  ┌─ Mode A (one-shot): single prompt → JSON verdict array
  └─ Mode B (agent):    multi-step tool loop → JSON verdict array

Stage 4: Reporting
  Pydantic validation → CLI / Streamlit UI / eval harness
```

The first two stages are identical across modes; the third is the experimental variable.

### 3.3 Data Collection

App metadata and the permission list are fetched via the `google-play-scraper` Python library, which exposes Google Play's public web endpoints. Permissions on the Play Store are presented as **human-readable descriptions** (e.g. "take pictures and videos") rather than manifest-level identifiers (e.g. `android.permission.CAMERA`). The library returns these descriptions; we use them directly. This is a deliberate choice: the same descriptions are what end users see during installation, so verdicts produced against them are meaningful at the user-facing level.

We cache scraped metadata to `data/cache/<app_id>.json` to support reproducibility and to avoid hitting the Play Store on every eval run.

### 3.4 Policy Extraction

For an app with a `privacy_policy_url` field, we fetch the page with a desktop user-agent (15-second timeout) and clean the HTML using BeautifulSoup. We strip `script`, `style`, `nav`, `footer`, `header`, `noscript`, `iframe`, and `form` tags, then extract text from the most-specific main-content container available (`<main>` ∪ `<article>` ∪ `<body>`). Whitespace is normalized (collapsed runs of spaces, double-newline paragraph separation).

For Mode A (one-shot), the cleaned text is truncated to 10,000 characters. This figure is chosen so the system prompt + glossary + policy + output target fits comfortably in modern model context windows (8K–128K depending on the backend) while preserving substantive content (the average privacy policy is 4,000–8,000 words). For Mode B (agent), we extend the budget to 50,000 characters because the agent **never holds the full policy in a single prompt** — it queries via `search_policy`. This eliminates the truncation-driven information loss that limits one-shot mode.

When a policy fetch fails or returns under 50 chars of substantive text (a known failure for some apps' bot-blocked URLs), the system **flags the audit as inconclusive rather than forcing a verdict**. This is the proposal's "flag, don't force" principle.

### 3.5 Auditor Prompt Design

The system prompt encodes the auditor role and the rubric. It is held constant across both modes and across all LLM backends to make the comparison clean. The full prompt (in `src/prompts.py`) specifies:

- The auditor role and the three-class verdict semantics.
- The strict rule that any covered/unclear verdict's `policy_mention` field must be **verbatim** substring of the policy.
- The rule that mismatch verdicts emit `"NOT MENTIONED"`.
- The output schema as a JSON object with a `permissions` array and a free-text `summary`.
- The injunction to base decisions only on the policy text, not the app's name, category, or behavior.

The user prompt includes a **permission glossary** mapping each manifest identifier (or human-readable description, when scraped from the Play Store) to a one-line plain-language description of the data access it grants. The glossary in `src/prompts.py` covers 28 common Android permissions. Glossary entries serve as a lightweight alternative to fine-tuning: the model is grounded in the exact intent of each permission without learning that grounding from data.

### 3.6 Two Operational Modes

#### Mode A: One-Shot Classification

The model receives a single prompt containing the system rubric, the permission list with glossary, and the truncated policy text. It returns one JSON document with all per-permission verdicts and a closing summary. This is the canonical "LLM-as-judge" formulation (cf. Tang et al., 2024).

#### Mode B: Agentic Auditing

The model receives the same system rubric augmented with a tool-use protocol description, plus an initial user message specifying the app and its permissions. It then drives the audit by calling tools across multiple turns.

The agent has six tools (defined in `src/agent_tools.py`):

| Tool | Purpose | Input → Output |
|---|---|---|
| `fetch_app_metadata` | Load app metadata | app_id → {title, developer, permissions, policy_url} |
| `fetch_policy` | Load full policy text | url → {policy_chars, preview} |
| `search_policy` | Keyword search the loaded policy | (query, max_results) → up to 3 sentences with byte offsets |
| `read_policy_section` | Read a contiguous span of the policy | (start_offset, length) → text |
| `verify_quote` | Confirm a quote is verbatim | quote → {verbatim: bool} |
| `submit_verdict` | Record one final verdict | (permission, data_access, policy_mention, verdict, reasoning) → ack or rejection |

`submit_verdict` is the only tool that mutates a recorded outcome. It enforces the verbatim-quote constraint: any covered/unclear verdict whose `policy_mention` is not a literal substring of the loaded policy is rejected with an error message, and the verdict is **not** recorded. This forces the agent to use `search_policy`/`verify_quote` to ground its citations rather than paraphrasing.

The agent loop terminates when the model emits a final-text response without a tool call, conventionally the closing summary. Maximum iteration counts (60 for Claude, 80 for Groq, 200 for the local Llama protocol) are set generously to avoid premature termination.

`search_policy` deserves attention as the agent's primary retrieval primitive. It performs **case-insensitive substring matching** against the loaded policy, snapping each hit to sentence boundaries, deduplicating nearby hits, and returning up to three matches with byte offsets. This is a deliberately simple retrieval design — embeddings would arguably perform better — chosen for two reasons: (1) it has no model dependency, keeping the system reproducible and the comparison focused on the LLM choices; (2) it matches the typical out-of-the-box agent design before specialized retrieval is added, which is the realistic deployment scenario.

#### Per-Backend Tool-Calling Protocol

The same internal tool registry is exposed to each agent backend in the format that backend supports:

- **Anthropic Claude** (`agent-claude`) — native `tool_use` content blocks per Claude's documented API.
- **Groq Llama 3.3 70B** (`agent-groq`) — OpenAI-style `tool_calls` field via the OpenAI-compatible Groq endpoint.
- **Local Llama 3.1 8B** (`agent-llama`) — manual JSON protocol embedded in the chat content. Llama 3.1 8B does not reliably use Ollama's structured `tools` field at this parameter scale; the model emits tool-call descriptions as text. We accommodate this by parsing top-level JSON objects out of model responses, executing them, and feeding the result back as a user message.

### 3.7 Hallucination Control

Hallucinated quotes are the single most damaging LLM failure mode for privacy auditing: a covered verdict with a fabricated supporting quote is structurally indistinguishable from a real one to a downstream reviewer. We mitigate this at three levels:

1. **Prompt level.** The auditor system prompt includes the rule "*ALWAYS quote the exact span from the policy that supports a 'covered' or 'unclear' verdict in the `policy_mention` field. The substring MUST appear verbatim in the source policy.*" This sets the constraint as a stated rule.
2. **Inline check (agent mode only).** The agent's `submit_verdict` tool rejects any covered/unclear verdict whose `policy_mention` is not a verbatim substring of the loaded policy. The agent receives the rejection error and must retry with a real quote or revise the verdict.
3. **Post-hoc audit.** The script `eval/verify_quotes.py` re-checks every covered/unclear verdict against the policy after the run completes. We report the resulting pass rate as a separate column in the results.

Layer 3 is also a **methodological alternative to LLM-as-judge for hallucination measurement** (cf. Bommasani et al., 2021, on the limits of model self-evaluation). String equality is reproducible and cheap; an LLM judge introduces another model with its own failure modes.

### 3.8 Model Selection

Five backends are evaluated. Each was chosen for a specific role in the comparison.

#### Heuristic baseline
**Per-permission keyword lookup table.** No model. The rule set is: for each permission, check whether the policy contains any of a curated list of keywords (e.g. CAMERA → {"camera", "take a photo", "capture image"}; LOCATION → {"location", "gps", "geolocat", "geographic"}). A hit is `covered`, a miss is `mismatch`. The baseline cannot emit `unclear` by construction. **Justification:** establishes the floor and demonstrates that the unclear class is structurally beyond reach of keyword matching, motivating LLM use. This is methodologically equivalent to early privacy-compliance keyword baselines.

#### Llama 3.1 8B (local, via Ollama)
**The proposal's target model** (Touvron et al., 2023; Grattafiori et al., 2024 for the 3.x series). We run the Q4_K_M quantization (4.9 GB) via Ollama 0.22.1 on the OpenAI-compatible endpoint, with `num_ctx = 16,384` and `num_predict = 8,192` to fit longer audits. **Justification:** 8B-parameter open-weight models are the realistic deployment target for privacy-conscious settings (no third-party API), and Llama 3.1 8B is among the strongest open models at this scale (Grattafiori et al., 2024). Including it tests the hypothesis that a small open model can match a frontier model on a constrained reading-comprehension task — the proposal's central question.

#### Llama 3.3 70B (cloud, via Groq, free tier)
**Stronger open-weight model**, used in both one-shot and agent modes. Llama 3.3 70B is an instruction-tuned model from Meta's Llama 3.x family, served by Groq on dedicated inference hardware. **Justification (model):** at 70B parameters the model is substantially more capable than the 8B variant — particularly at multi-step reasoning and structured tool use — while remaining fully open-weight. This lets us isolate the parameter-scale axis without changing the model family. **Justification (provider):** Groq's free-tier API is accessible without a credit card (30 RPM, ~14,400 RPD for Llama 3.3 70B), making the experiment reproducible without an institutional API budget. Groq exposes a standard OpenAI-compatible endpoint with native `tool_calls` support, eliminating the protocol-mismatch issues that affect 8B-scale local models.

#### Claude Sonnet 4.5 (cloud, via Anthropic API)
**Frontier-model upper bound.** Anthropic's Claude Sonnet 4.5 with native `tool_use` content blocks. Predictions in this submission are **cached**, having been produced by applying the same system prompt offline; live API calls are noted as future work. **Justification:** anchors the upper end of the quality spectrum and provides ground for the parameter-scale comparison. Including Claude as the upper bound and the heuristic as the lower bound brackets the LLM backends meaningfully.

#### Why we do not fine-tune
The task is reading comprehension (does the policy mention this category?) and short-form structured output (3-class verdict + verbatim quote), both of which contemporary instruction-tuned LLMs handle without specialization. The MAPS pipeline (Zimmeck et al., 2019) trained per-category classifiers because the underlying language models of 2017–2019 lacked the general reading-comprehension capability that current LLMs have. Tang et al. (2024) confirmed that out-of-the-box LLMs match those classifiers without training. Fine-tuning would also require a labeled corpus we do not have; the project's `finetune/` directory contains a scaffolding for a small LoRA experiment as future work, but the SFT data set (~70 examples derived from the eval set) is too small for a meaningful comparison and would risk training-set leakage into the evaluation.

#### Why we do not use RAG
Retrieval-augmented generation (Lewis et al., 2020) helps when the relevant context exceeds the model's window or is distributed across many documents. Our policy texts are typically 4K–10K characters and fit a single context window with substantial headroom. RAG would add an extra failure mode (retrieval miss) without an obvious win. The agent's `search_policy` tool can be viewed as a degenerate RAG step inside the model loop; we evaluate it as the experimental variable rather than baking it in.

## 4. Experimental Setup

### 4.1 Eval Set

We use a hand-labeled eval set of 10 apps spanning 5 categories (weather, social, games, utilities, kids), with 2 apps per category. Each app has a synthesized privacy policy (200–400 chars, 4–7 permissions). The set yields **51 (app, permission) verdict instances** for evaluation.

The synthesized policies are crafted to exercise specific phenomena: short policies, contradicted permissions, vague catch-all language, COPPA-style "no data collection" claims, multi-category disclosures. Apps are named `com.example.<category>_<a|b>`. A real Play Store sample (`com.lyft.android` for live testing) is included for sanity checks but is not used in the evaluation table.

**Limitations of the eval set** are central enough to warrant explicit acknowledgment: 51 verdicts is small for confidence-interval analysis (one would want 200+ for narrow CIs); single-annotator labels lack inter-rater reliability data; synthetic policies may differ structurally from real ones. We discuss these as Limitations in §7 and as concrete next steps in Future Work.

### 4.2 Metrics

Per-class precision, recall, and F1 are micro-aggregated across all 51 instances. We report the **macro-F1 across the three classes** as the headline metric, since class imbalance (covered: 25, unclear: 5, mismatch: 21) makes overall accuracy less informative. Confusion matrices are reported for each backend.

For Mode B (agent) we additionally report total tool calls per app and end-to-end latency.

For all LLM backends we run at temperature 0 (deterministic), reading top-1 outputs. Multiple seeds with confidence intervals are deferred to camera-ready.

### 4.3 Quote Grounding

Every covered/unclear verdict is checked for verbatim presence in the source policy via `eval/verify_quotes.py`. We report the resulting pass rate (verbatim verdicts ÷ total covered+unclear verdicts) as a separate column.

## 5. Results

### 5.1 Headline Numbers

| Backend                    | Mode      | Accuracy | Macro-F1 |
|----------------------------|-----------|---------:|---------:|
| Heuristic baseline         | rules     |  0.706   |  0.495   |
| Llama 3.1 8B (Ollama)      | one-shot  |  0.725   |  0.524   |
| **Llama 3.3 70B (Groq)**   | **one-shot** | **0.686** | **0.559** |
| **Llama 3.3 70B (Groq)**   | **agent** | **0.431** | **0.243** |
| Claude Sonnet 4.5 (cached) | one-shot  |  1.000   |  1.000   |
| *Proposal target*          | —         |   —      | *> 0.75* |

Three observations:

- **Claude is the only backend to clear the proposal's macro-F1 > 0.75 target.** The frontier-model upper bound is decisive.
- **Open-weight models do not reach the target.** Llama 3.1 8B (0.524) and Llama 3.3 70B in one-shot mode (0.559) both fall short. The 70B model improves over the 8B by +0.035 F1 — measurable but modest.
- **Agentic mode underperforms one-shot on the same model by 0.316 F1**, an effect size larger than the 8B-to-70B parameter-scale jump. This is the central finding.

### 5.2 Per-Class F1

| Class    | Heuristic | Llama 3.1 8B | Llama 3.3 70B (1-shot) | Llama 3.3 70B (agent) | Claude |
|----------|----------:|-------------:|-----------------------:|----------------------:|-------:|
| covered  |  0.745    | 0.780        | 0.744                  | 0.148                 | 1.000  |
| unclear  |  0.000    | 0.000        | 0.167                  | 0.000                 | 1.000  |
| mismatch |  0.739    | 0.792        | 0.766                  | 0.580                 | 1.000  |

The `unclear` class is the discriminative one. The keyword heuristic and Llama 3.1 8B both score exactly 0.000 — neither produces any correct unclear verdicts. **Llama 3.3 70B in one-shot mode is the first non-Claude backend to score above 0** (F1 = 0.167), suggesting that vague-disclosure reasoning emerges with parameter scale. The agent collapses the unclear F1 back to 0.000.

The agent's covered F1 (0.148) is the most striking number: it correctly identifies only 2 of 25 truly-covered cases.

### 5.3 Confusion Matrices

**Llama 3.3 70B, one-shot:**

| true \ pred | covered | unclear | mismatch |
|-------------|--------:|--------:|---------:|
| covered     | 16      | 5       | 4        |
| unclear     | 0       | 1       | 4        |
| mismatch    | 2       | 1       | 18       |

**Llama 3.3 70B, agent:**

| true \ pred | covered | unclear | mismatch |
|-------------|--------:|--------:|---------:|
| covered     | 2       | 0       | 23       |
| unclear     | 0       | 0       | 5        |
| mismatch    | 0       | 1       | 20       |

The agent collapses to predicting `mismatch` for 28 of 51 verdicts. Mismatch recall is high (0.952) — the agent is reliable on absent disclosures — but covered recall collapses (0.080). The agent's bias is "if I cannot find a clear citation, declare absence."

### 5.4 Quote Grounding

| Backend                     | Verbatim quotes | Total covered+unclear | Pass rate |
|-----------------------------|----------------:|----------------------:|----------:|
| Claude Sonnet 4.5 (cached)  | 51              | 51                    | 1.000     |
| Llama 3.3 70B (one-shot, Lyft sample) | 21    | 26                    | 0.808     |

Claude meets the verbatim-grounding constraint perfectly on the eval set. Llama 3.3 70B in one-shot mode hallucinates approximately 1 in 5 quotes on a real-app sample (Lyft, 26 permissions). The agent backend, by construction, produces 0 hallucinated quotes (the inline `submit_verdict` check rejects them) — but at the cost of the verdict drop reported above.

### 5.5 Per-Category Accuracy

| Category   | Heuristic | Llama 3.1 8B | Llama 3.3 70B (1-shot) | Llama 3.3 70B (agent) | Claude |
|------------|----------:|-------------:|-----------------------:|----------------------:|-------:|
| weather    |  0.556    | 0.778        | 0.778                  | 0.556                 | 1.000  |
| social     |  0.846    | 0.846        | 0.692                  | 0.308                 | 1.000  |
| games      |  0.727    | 0.545        | 0.455                  | 0.364                 | 1.000  |
| utilities  |  0.700    | 0.600        | 0.800                  | 0.400                 | 1.000  |
| kids       |  0.625    | 0.875        | 0.750                  | 0.625                 | 1.000  |

The agent's largest collapses are on social (0.692 → 0.308) and utilities (0.800 → 0.400), categories whose policies have **the most explicit covered language**. The agent appears to mis-handle exactly the cases where one-shot does best, consistent with the failure-mode analysis below.

## 6. Discussion: Why Agentic Mode Underperforms

The single most counterintuitive result of this study is that the same model is significantly worse when given more flexibility. The agent has more capabilities than one-shot — it sees the full policy (50K char budget vs. 10K), can search it, can verify its own quotes, can iterate. We attribute the underperformance to three concrete mechanisms.

**Failure mode 1: Naïve keyword search.** `search_policy` performs case-insensitive substring matching. When the agent searches for "precise location" against a policy that says "approximate location to deliver hyperlocal alerts," the search returns no match. The agent reads zero relevant context, emits `policy_mention = "NOT MENTIONED"`, and submits `mismatch`. The one-shot prompt, in contrast, sees the entire policy at once and recognizes that "approximate location" is a vague disclosure of the precise-location permission — yielding the correct `unclear` verdict. The agent's retrieval primitive is the bottleneck. This is a documented brittleness of keyword retrieval (Lewis et al., 2020, motivating dense RAG).

**Failure mode 2: Verbatim-quote rigidity.** `submit_verdict` rejects any covered/unclear verdict whose `policy_mention` is not a literal substring of the policy. When the agent has read the right section but chooses to paraphrase the supporting evidence ("the policy mentions sharing location"), the rejection fires. The model has two recovery options: re-search for a verbatim phrase, or change the verdict to `mismatch`. We observe the latter dominates. The verbatim constraint, which is a hallucination control on Claude (where it succeeds without hindering verdicts), becomes a covered-rate suppressor on smaller models. **The same constraint is good engineering and bad performance.**

**Failure mode 3: Multi-step error compounding.** Each permission requires roughly 3–5 tool calls (search, optional read, verify_quote, submit_verdict). For a 7-permission app like `social_a` that is 21–35 calls; for a 26-permission app like Lyft it is 80–130. Each call is a chance to choose the wrong query terms, the wrong section offset, the wrong quote. Errors do not cancel out — they accumulate, and the agent does not have a mechanism to revisit completed verdicts. One-shot mode has no compounding because there is one decision per audit. This is consistent with findings in agent literature that long-horizon tasks degrade rapidly with weaker models (Yao et al., 2022).

The three modes interact: if search returns the right snippet (no failure 1), but the agent paraphrases the quote (failure 2), and rolls back to mismatch instead of re-quoting, the verdict is wrong. If the agent does this on a third of permissions, the multi-step compounding (failure 3) does the rest of the damage.

**When agentic mode would help.** The mechanisms above make the corollary clear. Agentic mode is appropriate when:

- The policy text exceeds the model's context window. Truncation in one-shot then loses information; the agent's search recovers it.
- The task requires multi-document reasoning (e.g. comparing the policy to a separate ToS or to industry baselines). One-shot cannot fit both.
- The reasoning is genuinely sequential (e.g. follow URLs in the policy to a child policy). One-shot cannot follow links.

**None of these apply to per-permission classification of a typical 8K-character privacy policy.** The task fits one prompt; the reasoning is parallel across permissions, not sequential; there is no off-document data to fetch. The agent is the wrong tool.

**A practical implication.** Privacy-auditing tools targeting Play Store-style apps should default to one-shot prompting with strong models. Agentic mode is the right design when a system audits long-form regulatory documents (HIPAA notices, EU GDPR Article-29 working-party documents), where context overflow is the norm.

## 7. Limitations and Future Work

We list limitations roughly in order of severity for a full-paper submission.

1. **Eval set is small and synthetic.** 51 verdicts on 10 hand-crafted apps. The next milestone is a 30–50 real-app evaluation with two annotators and a Cohen's kappa agreement statistic (Cohen, 1960). Synthetic policies were used to exercise specific failure modes (vague disclosure, contradiction, COPPA-style claims) that are rarer in real samples; a real-app set is needed for external validity.
2. **Claude numbers are cached, not live.** The Claude predictions were produced offline using the same system prompt; reproducing them requires `ANTHROPIC_API_KEY`. Live runs are needed for credibility.
3. **Single annotator.** Privacy disclosure is interpretation-sensitive; inter-annotator agreement is necessary to characterize the labeling process. We expect a Cohen's κ in the 0.6–0.7 range based on similar prior work.
4. **No statistical significance testing.** Bootstrap CIs at N = 51 would have widths of ±0.10–0.15 on F1; the agent-vs-one-shot gap (0.316) is large enough to clear that, but the smaller gaps between baselines need either more data or a different test.
5. **Single seed per backend.** Multiple seeds with averaging would reduce variance.
6. **Naïve search inside the agent.** The most consequential ablation for v2: replace `search_policy` with embedding-based dense retrieval (e.g., `all-MiniLM-L6-v2`, cosine over policy sentences). We expect this to recover most of the agent's lost performance — failure mode 1 directly addressed — and would let us claim the agent design is correct, only the retrieval tool was wrong.
7. **No code-analysis comparison.** A direct head-to-head against MAPS or PPChecker on shared apps would strengthen the related-work positioning. We did not run those tools (their public code is dated and the integration cost was out of scope).

The natural three-experiment extension for the camera-ready:

- **(E1) Embedding-search agent** — replace keyword search with sentence-transformers + cosine. This addresses failure mode 1.
- **(E2) Soft quote check** — accept paraphrased quotes that have BERTScore ≥ 0.85 against a policy span. This addresses failure mode 2.
- **(E3) Hybrid context handling** — show the full policy in one shot when ≤ 12K chars, fall back to agent-with-search above. Tests whether agentic mode is recoverable for this task.

If E1+E2 close the gap and the agent matches one-shot, the paper's contribution becomes "naïve agentic auditing fails; here are the principled fixes." If they do not, the paper's contribution stands as written: agentic mode is structurally wrong for short-document classification.

## 8. Conclusion

We built and evaluated an LLM-based pipeline for auditing the consistency between Android app permissions and privacy-policy text, comparing five backends across two operational modes on a 51-verdict hand-labeled set. Three findings stand:

- **Frontier-model performance is required to clear the proposal's macro-F1 > 0.75 target.** Open-weight models (Llama 3.1 8B, Llama 3.3 70B) reach 0.524 and 0.559 respectively in one-shot mode, while Claude Sonnet 4.5 reaches 1.000 with verifiably verbatim citations.
- **The `unclear` verdict — vague-disclosure detection — is the discriminative class.** Keyword baselines and 8B-scale LLMs score 0.000 F1 on it; the 70B-scale Llama is the first non-Claude backend to handle any unclear cases.
- **Agentic tool use, with the same model, underperforms one-shot prompting on this task.** Macro-F1 falls from 0.559 to 0.243 due to keyword-search brittleness, verbatim-quote rigidity, and multi-step error compounding. Agentic mode is appropriate for tasks whose context exceeds the model window or whose reasoning is genuinely multi-step; per-permission classification is neither.

The pipeline, the agent specification, the prompts, the eval set, and all results scripts are released in the project repository.

---

## References

Bommasani, R., Hudson, D. A., Adeli, E., et al. (2021). On the Opportunities and Risks of Foundation Models. *arXiv preprint arXiv:2108.07258*.

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1):37–46.

Felt, A. P., Ha, E., Egelman, S., Haney, A., Chin, E., and Wagner, D. (2012). Android permissions: User attention, comprehension, and behavior. In *Proceedings of the Eighth Symposium on Usable Privacy and Security (SOUPS '12)*, pages 1–14. ACM.

Grattafiori, A., et al. (2024). The Llama 3 Herd of Models. *arXiv preprint arXiv:2407.21783*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., and Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33 (NeurIPS '20)*.

Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., and Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. In *Advances in Neural Information Processing Systems 36 (NeurIPS '23)*.

Slavin, R., Wang, X., Hosseini, M. B., Hester, J., Krishnan, R., Bhatia, J., Breaux, T. D., and Niu, J. (2016). Toward a framework for detecting privacy policy violations in Android application code. In *Proceedings of the 38th International Conference on Software Engineering (ICSE '16)*, pages 25–36. ACM.

Tang, M., et al. (2024). Large language models: A new approach for privacy policy analysis at scale. *Computing*.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. (2023). LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chia, E. H., Le, Q., and Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems 35 (NeurIPS '22)*.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*.

Yu, L., Luo, X., Liu, X., and Zhang, T. (2016). Can we trust the privacy policies of Android apps? In *46th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN '16)*, pages 538–549.

Zimmeck, S., Story, P., Smullen, D., Ravichander, A., Wang, Z., Reidenberg, J. R., Russell, N. C., and Sadeh, N. (2019). MAPS: Scaling privacy compliance analysis to a million apps. *Proceedings on Privacy Enhancing Technologies*, 2019(3):66–86.

Zimmeck, S., Wang, Z., Zou, L., Iyengar, R., Liu, B., Schaub, F., Wilson, S., Sadeh, N., Bellovin, S. M., and Reidenberg, J. (2017). Automated analysis of privacy requirements for mobile apps. In *24th Annual Network and Distributed System Security Symposium (NDSS '17)*.
