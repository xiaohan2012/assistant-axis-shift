# assistant-axis-shift

How the assistant axis shifts across Olmo 3 7B Instruct's post-training stages (SFT, DPO, RLVR).
Code, data pointer and write-up for the LessWrong post ([`docs/writeup/lesswrong.md`](docs/writeup/lesswrong.md)).

Method: [Lu et al. (2025)](https://arxiv.org/abs/2601.10387), run through a fork of their [assistant-axis](https://github.com/safety-research/assistant-axis) code with Olmo 3 support.

## Setup

```bash
git clone --recurse-submodules https://github.com/xiaohan2012/assistant-axis-shift.git
cd assistant-axis-shift
uv sync
```

- **Submodule:** `external/assistant-axis` is the fork.
- **Keys:** copy `.env.example` to `.env` (judge and Runpod keys, only needed for the pipeline).

## Layout

- **`notebooks/`** (marimo), the post's results:
  - `assistant_axis.py`: role projections on each stage's assistant axis.
  - `pass_rate.py`: tendency to role play per stage.
  - `q1_robustness.py`: split-half, label shuffling and matched-sample checks.
- **`src/persona_shift/`:** loading, projection, chart and robustness code.
- **`configs/`:** role list, desirability labels, labeling and judge prompts.
- **`scripts/`:** extraction pipeline on a Runpod GPU pod ([`scripts/README.md`](scripts/README.md)).
- **`docs/writeup/`:** the post and its figures.

Open a notebook: `uv run marimo edit --watch notebooks/<name>.py`.

## Data

On Hugging Face: [han-xiao/assistant-axis-shift](https://huggingface.co/datasets/han-xiao/assistant-axis-shift), about 18 GB.

```bash
hf download han-xiao/assistant-axis-shift --repo-type dataset --local-dir data/phase-4
```

Notebooks read `data/phase-4/{sft,dpo,rlvr}-sysprompt/`, one folder per checkpoint:

- **`vectors-glm-5.3-flash/`, `axis-glm-5.3-flash.pt`:** per-role vectors and the assistant axis, for `assistant_axis.py`.
- **`scores-glm-5.3-flash/`:** judge scores, for `pass_rate.py`.
- **`activations-layer16/`:** per-response layer-16 activations (`keys`, `acts` per role), for `q1_robustness.py`.
  - A layer-16 slice of the pipeline's `activations/`; the slicing script is not in this repo yet.

**Regenerate:** run `scripts/pod/run_chain.sh` on a GPU pod.

- Checkpoints: `allenai/Olmo-3-7B-Instruct-SFT`, `allenai/Olmo-3-7B-Instruct-DPO`, `allenai/Olmo-3-7B-Instruct`.
- Role lists: `configs/phase-4/roles_275.json` and `configs/phase-4/roles_default.json`.
