# NLP project

By group 19

- 2605141, Duc Quan Nguyen
- 2305809, Tino Vilmar Saarenpää

## Project overview

[main.ipynb](main.ipynb) implements clinical-note summarization using four
agents: information extraction, drafting, factual verification, and revision.
It includes a single-agent baseline, automatic evaluation, an extraction
ablation, resource measurements, and charts.

The dataset is `starmpcc/Asclepius-Synthetic-Clinical-Notes` from Hugging Face.
Step 1 cleans the dataset and selects 100 notes with `random_state=67`.
Saved case IDs follow this sample order, not the original dataset indices.
This project uses synthetic clinical notes for educational purposes and is
not intended for clinical decisions.

## Requirements and library versions

Use VS Code with the **Python** and **Jupyter** extensions. Choose one of these
separate environments:

| Environment | Python | PyTorch | Dependency file |
|---|---|---|---|
| Original CPU environment | 3.13.14 | 2.14.1 (CPU build in the recorded environment) | [requirements.txt](requirements.txt) |
| AMD GPU, Windows 11 | 3.12.10 | 2.9.1+rocm7.2.1 | [requirements-amd.txt](requirements-amd.txt) |

Both files record exact versions of the notebook libraries, including
Transformers 5.20.1, datasets 5.1.0, pandas 3.0.6, Matplotlib 3.11.2,
NLTK 3.9.2, BERTScore 0.3.13, ROUGE Score 0.1.2, rank-bm25 0.2.2, and
Accelerate 1.15.0. They pin direct dependencies and kernel support, not every
transitive dependency.

The separate [pyproject.toml](pyproject.toml) declares Python >=3.14 and some
different dependency minimums. Use the requirements-file workflow below to
reproduce the recorded environments; do not combine it with `pip install .`.

### Option A: CPU environment

From the repository root in PowerShell, with Python 3.13 installed:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check that `python --version` reports Python 3.13 before creating the environment.
If it points to another version, use the full path to your Python 3.13 executable.
In VS Code, select `.venv\Scripts\python.exe` as the notebook kernel.

### Option B: AMD GPU environment

This configuration was checked on an **AMD Ryzen AI 9 365 / Radeon 880M** under
Windows 11. GPU detection and a small GPU tensor operation passed, and the user
confirmed that all four models loaded onto `cuda:0`. A complete GPU experiment
is not assumed to be validated by those checks.

Install Python 3.12 and meet the driver prerequisites in
[AMD's official Ryzen APU guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html).
Then run, adjusting the interpreter path if necessary:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv-amd
.\.venv-amd\Scripts\python.exe -m pip install -r requirements-amd.txt
.\.venv-amd\Scripts\python.exe -m ipykernel install --user --name nlp-project-amd --display-name "NLP Project (AMD GPU, Python 3.12)"
```

Select **NLP Project (AMD GPU, Python 3.12)** or
`.venv-amd\Scripts\python.exe` as the VS Code notebook kernel. Do not install
the CPU requirements into this environment, as they specify a different
PyTorch build.

Verify GPU access:

```powershell
.\.venv-amd\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.hip); print('GPU available:', torch.cuda.is_available())"
```

AMD ROCm uses the `torch.cuda` API, so `cuda:0` is expected even on an AMD GPU.
The notebook uses `device_map="auto"`; models may be offloaded to CPU/disk if
memory is insufficient. Test generation on one note before starting a large
experiment. Switching kernels requires rerunning imports and model loading.

## How to run

Open [main.ipynb](main.ipynb) from the repository root. Keep that folder as the
notebook working directory, because file paths are relative to it.
[dictionary.txt](dictionary.txt) is required for abbreviation expansion.

For a fresh experiment, execute cells in order:

| Section | Purpose |
|---|---|
| Setup and Step 1 | Import libraries, initialize agents, and load/sample the dataset |
| Steps 2-3 | Extract information and generate the single-agent baseline |
| Step 4 | Build/load the BM25 retrieval index and demonstrate retrieval |
| Steps 5-6 | Define/run drafting, verification, revision, and save case results |
| Step 7 | Calculate/save ROUGE, METEOR, BERTScore, and length statistics |
| Step 8 | Review example cases and qualitative observations |
| Step 9 | Compare full pipeline versus no extraction and measure resource use |
| Step 10 | Plot saved evaluation and verification results |
| Step 11 | Discuss findings and limitations |

The agent models are:

- Extraction: `Qwen/Qwen3-0.6B` (also used by the current baseline call).
- Drafting: `HuggingFaceTB/SmolLM2-1.7B-Instruct`.
- Verification: `LiquidAI/LFM2.5-1.2B-Instruct`.
- Revision: `ibm-granite/granite-3.3-2b-instruct`.

Verification splits drafts into statements and checks them against the source
note. The Step 4 retrieval demonstration is separate from the current verifier.

Internet access is needed for the first dataset/model download, the
`distilbert/distilbert-base-uncased` BERTScore model, and NLTK WordNet.
Full inference can take substantial time and memory, particularly on CPU.

### Reuse saved results

To inspect existing results, open the notebook's saved outputs without running
inference. To regenerate charts, run the initial imports and JSON helper cell,
then the relevant Step 10 definitions/cells in section order. Model
initialization and Steps 2-6 are not required for plotting existing JSON.
Step 9.2 also only loads saved data into a table.

**Do not use Run All just to view results.** Generation cells are active and
replace existing result files. Back up files you want to retain first.

### Change the Step 9 experiment size

Edit the run cell:

```python
STEP9_NUM_NOTES = 20
step9_ablation = step9_run_ablation(
    sample, agent1, agent2, agent3, agent4, num_notes=STEP9_NUM_NOTES,
)
```

Choose a positive integer no greater than `len(sample)` (currently 100).
The first N notes run through both variants, producing **2N ablation records**
and **N full-pipeline verification measurements** for Step 10.5. Rerun the
function-definition cell after editing its code.

The runner checkpoints completed variants, but does not resume a partial run:
starting it again replaces the previous Step 9 files. Both variants are
multi-agent pipelines, not a single-agent cost comparison. Use the same
hardware/environment for both. Step 9.3 contains static observations and must
be updated after changing the measurements; rerunning 9.2 alone does not run
a new experiment.

## Outputs and caches

- `.cache`: downloaded agent/evaluation models; NLTK resources under
  `.cache\nltk`. The dataset loader uses the Hugging Face datasets cache
  unless configured separately.
- [intermediate_states/](intermediate_states/): saved JSON outputs, scores,
  resource measurements, and the BM25 pickle index. Only load trusted pickles.
- Step 7 saves evaluation results as `step7_*.json`.
- Step 9 saves `step9_ablation.json` and `step9_compute_usage.json`.
- Step 10.3 saves `step10_verification_flags.json`; Step 10.4 uses those flags.
- Step 10.5 reads `step10_verification_usage.json`, saved by Step 9.
  `step10_verification_demo.json` contains dummy preview data, not measurements.
- [pipeline-diagram.svg](pipeline-diagram.svg) is the diagram embedded in Step 10.1.

Similarity scores and bracketed verification-label counts do not independently
establish factual accuracy. Do not interpret shorter output or lower resource
use as proof of better summaries.
