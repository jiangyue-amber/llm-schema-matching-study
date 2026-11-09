# Extending LLM-Based Schema Matching to Multiple Database Pairs

This project extends the framework from **Schema Matching with Large Language Models: an Experimental Study** beyond its MIMIC-IV–OMOP database pair. It adapts additional schema pairs from **LLMATCH** to the original framework and evaluates **Gemini 2.5 Flash** and **Qwen3-1.7B** across eight database pairs, with three runs for each model–pair combination.

The repository contains schema preparation scripts, processed schema descriptions, ground-truth mappings, model-specific matching code, and experimental results.

## Background and sources

- **Original framework:** [Schema Matching with Large Language Models: an Experimental Study](https://arxiv.org/abs/2407.11852). The [original artifact repository](https://github.com/UHasselt-DSI-Data-Systems-Lab/code-schema-matching-LLMs-artefacs) provides the starting point for this project; the matching implementation originates from [fm_matcher](https://github.com/UHasselt-DSI-Data-Systems-Lab/code-demo-fm-matcher).
- **Additional benchmark data:** [LLMATCH: A Unified Schema Matching Framework with Large Language Models](https://arxiv.org/abs/2507.10897) and its [source repository](https://github.com/knowledge-fusion/LLMatch). This project processes its schema data for use with the original paper's matching framework.

The extensions here include adapting the additional database pairs, preparing compatible table and attribute descriptions, adding Gemini and local Hugging Face model backends, and collecting repeated experimental results. The matching experiments use the original framework's **1-to-N prompting approach**.

## Database pairs

The eight pairs include the original MIMIC-IV–OMOP pair and seven additional pairs adapted from LLMATCH.

| Source → target | Ground truth in `benchmark/` | Processed descriptions | Result label |
| --- | --- | --- | --- |
| MIMIC-IV → OMOP | `ground_truth.csv` | `schema_documentations_original/` | `original` |
| MIMIC-III → OMOP | `MIMIC_III-OMOP-ground_truth.csv` | `schema_documentations_mimic/` | `mimic` |
| CMS → OMOP | `CMS-OMOP-ground_truth.csv` | `schema_documentations_cms/` | `cms` |
| CPRD Aurum → OMOP | `CPRD_AURUM-OMOP-ground_truth.csv` | `schema_documentations_cprda/` | `cprda` |
| CPRD GOLD → OMOP | `CPRD_GOLD-OMOP-ground_truth.csv` | `schema_documentations_cprdg/` | `cprdg` |
| Synthea → OMOP | `SYNTHEA-OMOP-ground_truth.csv` | `schema_documentations_synthea/` | `synthea` |
| IMDb → Sakila | `IMDB-SAKILA-ground_truth.csv` | `schema_documentations_imdb/` | `imdb` |
| Bank1 → Bank2 | `BANK1-BANK2-ground_truth.csv` | `schema_documentations_bank/` | `bank` |

## Matching workflow

1. Convert schema metadata into text descriptions for tables and attributes.
2. Load a pair's ground-truth mappings and obtain the distinct source–target table combinations represented in those mappings.
3. For each table combination, build 1-to-N prompts: one source attribute is compared with the attributes of a target table, using names and descriptions.
4. Query the selected model and parse its answers into `yes`, `no`, or `unknown` decisions for attribute pairs.
5. Export decisions to CSV, join them with the ground truth, and calculate evaluation metrics.

**Evaluation scope:** table combinations are selected using the ground truth. These experiments evaluate attribute matching within those combinations, rather than discovery of matching tables across an entire database. Exported attribute pairs absent from the ground truth are labeled `False`.

## Models and repeated runs

| Model | Model identifier | Execution |
| --- | --- | --- |
| Gemini 2.5 Flash | `gemini-2.5-flash` | Gemini API |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | Local inference through Hugging Face Transformers |

The current notebooks set temperature to `1.0` and request one answer per prompt (`GEMINI_N=1` / `HUGGINGFACE_N=1`). Three experiment runs are stored separately in `results1/`, `results2/`, and `results3/`.

## Repository layout

| Path | Description |
| --- | --- |
| `LLMATCH schemas/` | Schema inputs and extraction scripts |
| `benchmark/` | Ground-truth mappings for eight pairs |
| `schema_documentations_*/` | Processed descriptions for each pair |
| `fm_matcher/utils_gemini/` | Gemini matching backend |
| `fm_matcher/utils_huggingface/` | Local Qwen matching backend |
| `fm_matcher/utils/` | Original matching implementation |
| `templates/` | Prompt templates |
| `generate_gemini_results_1-N.ipynb` | Gemini experiment notebook |
| `generate_huggingface_results_1-N.ipynb` | Qwen experiment notebook |
| `results1/`, `results2/`, `results3/` | CSV results from the three runs |
| `evaluate_metrics.py` | Evaluation script |
| `evaluation_metrics_results*.csv` | Per-run metric summaries |
| `experiment_results.xlsx` | Experiment results workbook |

Each run directory contains eight `gemini_<pair>_results_1-N.csv` files and eight `huggingface_<pair>_results_1.7b.csv` files. SQLite files store local experiment artifacts and cached answers.

## Setup

Use Python 3.11 as specified by the matching package. From the repository root, create an environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r fm_matcher/requirements.txt
pip install jupyterlab google-generativeai tenacity torch 'transformers>=4.51.0' accelerate
```

These are starting-point installation commands, not a fully pinned reproduction environment. The root requirements file does not include all dependencies used by the new backends.

The Qwen CUDA path explicitly requests FlashAttention 2. Install a compatible `flash-attn` build for your PyTorch/CUDA environment before using that path. Batch size and concurrency should be adjusted to available GPU memory.

## Running an experiment

Choose `generate_gemini_results_1-N.ipynb` or `generate_huggingface_results_1-N.ipynb`. The notebooks are currently configured for CMS–OMOP and require manual edits to switch pairs and runs.

For Gemini, set `GEMINI_API_KEY` in the environment or enter it at the notebook's secure prompt. Adjust request limits to your account. Executing the notebook sends API requests and may incur charges.

For Qwen, configure `GPU_INDEX` before importing PyTorch and tune `HUGGINGFACE_BATCH_SIZE` and `HUGGINGFACE_MAX_CONCURRENT` for your hardware.

### Preparing additional schemas

`LLMATCH schemas/extract_schema_info.py` extracts table and column comments from SQL files. `extract_schema_info_json.py` extracts descriptions from JSON schema metadata. Their input paths, output directories, and schema prefixes are hard-coded and must be updated for the desired schema.

The processed descriptions use these naming conventions:

```text
<schema>-table-<table>.txt
<schema>-<table>-<attribute>.txt
```

## Evaluating saved results

Run:

```ba
python evaluate_metrics.py
```
For each result file, it reports unknown percentage, precision, recall, F1, and TP/FP/FN/TN counts. `unknown` decisions are counted separately for the unknown percentage and converted to `no` for classification metrics. Model-level summaries are arithmetic means of file-level metrics. The printed headings still say “7 datasets,” although each main run directory and saved metric summary contains eight pairs per model.

The three saved `evaluation_metrics_results*.csv` files contain the per-run summaries. This script does not aggregate metrics across the three runs.

## License

The original MIT license and author attribution are retained in [LICENSE](LICENSE). See the upstream projects and dataset sources for the terms applicable to their respective materials.
