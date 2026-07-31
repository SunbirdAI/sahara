# Sahara Benchmark

This repository contains the code for running the Sahara benchmark across LLM models. The included evaluation launcher uses vLLM to evaluate a Hugging Face model on the complete set of Sahara tasks and saves the model generations for each task.

## Prerequisites

Before starting, make sure that you have:

- Git and [`uv`](https://docs.astral.sh/uv/) installed.
- A Linux machine with NVIDIA GPUs and a working CUDA driver.
- Access to the model that you want to evaluate on Hugging Face.
- Access to the gated [`UBC-NLP/sahara_benchmark`](https://huggingface.co/datasets/UBC-NLP/sahara_benchmark) dataset. Request access from its Hugging Face page before running the evaluation.

The supplied launcher is configured to expose GPUs `0,1,2,4`, use all visible GPUs for vLLM tensor parallelism, and store downloaded files in `/workspace/.cache`. If your machine has a different GPU layout or cache location, update `CUDA_VISIBLE_DEVICES` or `cache_dir` in `evaluation_scripts/evaluate_sahara.sh` before running it.

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/jim-junior/sahara.git
cd sahara
```

### 2. Create and activate an environment

Using a virtual environment keeps the benchmark dependencies isolated:

```bash
uv venv
source .venv/bin/activate
```

### 3. Install the dependencies

Run the following commands in the exact order shown:

```bash
uv pip install numpy pandas scikit-learn datasets evaluate tenacity \
    openai anthropic torch transformers httpx \
    huggingface_hub sacrebleu rouge_score bert_score seqeval editdistance bitsandbytes accelerate
```

Then:

```bash
uv pip install "datasets<3.0.0"
```

Then:

```bash
uv pip uninstall vllm torch torchvision torchaudio \
    xformers flash-attn triton
```

Then:

```bash
uv cache clean
```

Then:

```bash
uv pip install vllm --torch-backend=auto
```

### 4. Log in to Hugging Face

Lastly, authenticate with the Hugging Face account that has access to the benchmark dataset and the model:

```bash
huggingface-cli login
```

Paste your Hugging Face access token when prompted.

### 5. Run the benchmark

The evaluation script uses `Sunbird/Sunflower-Qwen3.5-9B` by default. Run it from the `evaluation_scripts` directory:

```bash
cd evaluation_scripts
./evaluate_sahara.sh
```

To evaluate a different Hugging Face model, pass its model ID with `-m`:

```bash
./evaluate_sahara.sh -m organization/model-name
```

The launcher evaluates these tasks:

- Text classification: `news`, `sentiment`, `topic`, `xlni`, and `lid`
- Text generation: `title`, `summary`, and `paraphrase`
- Machine translation: `mt_eng2xx`, `mt_fra2xx`, and `mt_xx2xx`
- Knowledge, reasoning, and question answering: `mmlu`, `mgsm`, `belebele`, and `squad_qa`
- Token-level tasks: `phrase`, `pos`, and `ner`

The full benchmark can take a long time and requires enough aggregate GPU memory to load the selected model with an 8,192-token context window. The launcher processes examples with a batch size of `1000`.

## Outputs

For a model ID such as `Sunbird/Sunflower-Qwen3.5-9B`, `/` and `-` are replaced with underscores when the output directory is created. Results are written relative to `evaluation_scripts/` as follows:

```text
outputs/
└── Sunbird_Sunflower_Qwen3.5_9B/
    ├── <task>_generation.json
    └── csv/
        └── <task>_generation.csv
```

Each `<task>_generation.json` file contains one JSON object per line with the language code, generated answer, and example ID. The CSV files additionally retain prompts and raw model generations for inspection.
