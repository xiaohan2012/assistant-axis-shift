# assistant-axis-shift

How the assistant axis shifts across the post-training stages of Olmo 3 7B Instruct (SFT, DPO, RLVR).
Code and write-up behind the LessWrong post in [`docs/writeup/lesswrong.md`](docs/writeup/lesswrong.md).

The method follows [Lu et al. (2025)](https://arxiv.org/abs/2601.10387), through a fork of their [assistant-axis](https://github.com/safety-research/assistant-axis) code with Olmo 3 support.

## Setup

```bash
git clone --recurse-submodules git@github.com:xiaohan2012/assistant-axis-shift.git
cd assistant-axis-shift
uv sync
```

`external/assistant-axis` is a git submodule pointing at the fork.
Copy `.env.example` to `.env` for the judge and Runpod keys.

## Layout

- `notebooks/`: marimo notebooks behind the post's results.
  - `assistant_axis.py`: role projections on each stage's assistant axis.
  - `pass_rate.py`: the model's tendency to role play per stage.
  - `q1_robustness.py`: split-half, label shuffling and matched-sample checks.
- `src/persona_shift/`: loading, projection, chart and robustness code used by the notebooks.
- `configs/`: role list, desirability labels, and the labeling and judge prompts.
- `scripts/`: the extraction pipeline on a Runpod GPU pod; see [`scripts/README.md`](scripts/README.md).
- `docs/writeup/`: the post and its figures.

Open a notebook with `uv run marimo edit --watch notebooks/<name>.py`.

## Data

The data is not shared yet.
The notebooks read `data/phase-4/{sft,dpo,rlvr}-sysprompt/`, one directory per checkpoint, which the pipeline writes:

- `vectors-glm-5.3-flash/` and `axis-glm-5.3-flash.pt`: per-role vectors and the assistant axis, used by `assistant_axis.py`.
- `scores-glm-5.3-flash/`: judge scores, used by `pass_rate.py`.
- `activations-layer16/`: per-response layer-16 activations, one file per role with `keys` and `acts`, used by `q1_robustness.py` to rebuild vectors from question halves. It is a layer-16 slice of the pipeline's `activations/`; the slicing script is not in this repo yet.

To regenerate it, run `scripts/pod/run_chain.sh` on a GPU pod for the three checkpoints (`allenai/Olmo-3-7B-Instruct-SFT`, `allenai/Olmo-3-7B-Instruct-DPO`, `allenai/Olmo-3-7B-Instruct`), with both `configs/phase-4/roles_275.json` and `configs/phase-4/roles_default.json`.
