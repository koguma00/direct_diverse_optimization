# Direct Diversity Optimization

Official training and evaluation code for Direct Diversity Optimization (DDO).

## Installation

```bash
conda env create -f environment.yml
conda activate ddo
python -m pip install -e .
```

Python 3.10 and CUDA are recommended. Model weights are downloaded from Hugging Face.

## Released artifacts

The trajectory evidence and released DDO checkpoints are distributed as two separate ZIP files. From the repository root, extract both archives in place:

```bash
unzip /path/to/direct_diverse_optimization_datasets.zip -d .
unzip /path/to/direct_diverse_optimization_ddo_checkpoints.zip -d .
```

The dataset archive supplies `trajectories/base/` and `trajectories/dtc/`. The checkpoint archive supplies nine DDO adapters and their nine reference adapters under `checkpoints/`: four BabyAI tasks, four BabaIsAI tasks, and WebShop. Other baseline adapters are not part of the public release. The archive does not include base-model weights, which are downloaded from Hugging Face. Method-specific training datasets and resumable trainer state are not included.

## Benchmarks

Clone the official repositories at the paths used by the code:

```bash
mkdir -p benchmarks
git clone https://github.com/balrog-ai/BALROG.git benchmarks/BALROG
git -C benchmarks/BALROG checkout b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5
git clone https://github.com/princeton-nlp/WebShop.git benchmarks/WebShop
git -C benchmarks/WebShop checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
```

BabyAI tasks are `goto`, `pickup`, `open`, and `comp`. BabaIsAI uses `basic` (paper task: *goto*), `room` (*two-room goto*), `stop` (*two-room break-stop goto*), and `flex` (*two-room optional break-stop goto*). WebShop uses `webshop`. The code converts these short names to the upstream environment IDs.

WebShop paper evaluation uses the full product catalog, human goals, and full
Lucene index. Install them with the setup script from the pinned official
checkout:

```bash
cd benchmarks/WebShop
bash setup.sh -d all
cd ../../..
```

This creates `data/items_shuffle.json`, `data/items_ins_v2.json`,
`data/items_human_ins.json`, and `search_engine/indexes/` where the DDO
evaluator expects them. The benchmark directories are ignored by Git.

## Training

Edit `configs/training.yaml`, then run:

```bash
python train.py --config configs/training.yaml
```

The config trains one benchmark task and one method. Supported methods are `base`, `reference`, `dpo`, `divfreq`, `divprob`, `tiedpo_rk`, `tiedpo_dav`, and `ddo`. `configs/training.yaml` lists the base-trajectory, DTC, dataset, SFT, preference-training, model, path, and runtime options. Its checked-in values are the paper defaults; stage-specific values override the shared `decoding` and `lora` defaults. Unknown fields fail validation instead of being ignored.

Reusable artifacts are separated by type:

```text
trajectories/base/<benchmark>/<task>/
trajectories/dtc/<benchmark>/<task>/
datasets/<benchmark>/<task>/<method>/
checkpoints/<benchmark>/<task>/<method>/
```

`start_from` controls which released artifacts are reused:

- `scratch`: create base trajectories, reference SFT, DTC, preference data, and the final checkpoint.
- `dtc`: read released base trajectories, DTC, and reference checkpoint; then build preference data and train.
- `dataset`: read released preference data and reference checkpoint; then train.

A portable DTC release keeps base and branch trajectories separate:

```text
trajectories/base/<benchmark>/<task>/**/*_llm_trace.json
trajectories/dtc/<benchmark>/<task>/_dtc/branch_index.jsonl
trajectories/dtc/<benchmark>/<task>/**/*__dtc_*_llm_trace.json
```

Reference SFT uses 10 epochs. Preference training uses 5 epochs for BabyAI and BabaIsAI and 15 epochs for WebShop. DTC uses `divergence_count: 5` (`K_d`) and `alt_budget: 3` (`K_a`); short trajectories and rejected alternatives can produce fewer retained branches.

Training state needed for checkpointing and resume is written under the ignored `.runs/` directory. Successful training copies only `adapter_model.safetensors` and `adapter_config.json` to `paths.output_checkpoint`. Progress is printed to the terminal.

Validate the config and resolved stage paths without starting an experiment:

```bash
python train.py --config configs/training.yaml --dry-run
```

## Evaluation

Released checkpoints contain only the PEFT files required for inference:

```text
checkpoints/<benchmark>/<task>/<method>/
├── adapter_model.safetensors
└── adapter_config.json
```

Edit `configs/eval.yaml` so that `checkpoint` points to the relative adapter directory and `results_dir` names the retained evaluation result directory, then run:

```bash
python eval.py --config configs/eval.yaml
```

Validate the config and paths without running inference:

```bash
python eval.py --config configs/eval.yaml --dry-run
```

`configs/eval.yaml` exposes rollout counts, seeds, decoding, WebShop success/class definitions, client timeouts, and managed-vLLM settings including dtype and tensor parallelism. The base model is downloaded from Hugging Face and is not included in a checkpoint bundle. `trajectories/`, `datasets/`, `checkpoints/`, `.runs/`, and `results/` are ignored by Git.

## Q&A

### Evaluation stops before producing a valid action. What should I try?

The paper default is `max_tokens: 8192`. If a model repeatedly reaches that limit before emitting an executable action, increase both `decoding.max_tokens` and `evaluation.max_tokens` in `configs/eval.yaml`. Keep `evaluation.server_max_model_len` large enough for the prompt and the increased generation budget. Changing this value changes the evaluation configuration, so report it with the resulting numbers.

This repository is a streamlined refactoring of the experimental code used for the paper.
