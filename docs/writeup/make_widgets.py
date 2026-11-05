"""Write the post's interactive charts as standalone HTML for LessWrong custom widgets.

Run from the repo root with the data in data/phase-4:

    uv run python docs/writeup/make_widgets.py

The widgets load Vega from the jsdelivr CDN, which LessWrong's widget sandbox allows.
"""

from pathlib import Path

import altair as alt
import pandas as pd

from persona_shift import persona_axes as pa
from persona_shift import persona_charts as pc

DATA_DIR = Path("data/phase-4")
OUT_DIR = Path("docs/writeup/widgets")
LAYER = 16


def stage_projections(vectors: pd.DataFrame, axes: pd.DataFrame) -> pd.DataFrame:
    """Each stage's roles projected on its own assistant axis, relative to its mean role vector."""
    layer_axes = axes[axes["layer"] == LAYER]
    rows = []
    for stage in ["SFT", "DPO", "RLVR"]:
        roles, matrix, default = pa.stage_matrix(vectors, stage, LAYER)
        axis = layer_axes.loc[layer_axes["stage"] == stage, "axis"].iloc[0]
        origin = matrix.mean(axis=0)
        rows += [
            {"stage": stage, "role": role, "is_default": False, "projection": float(p)}
            for role, p in zip(roles, pa.project(matrix, axis, origin))
        ]
        rows.append(
            {
                "stage": stage,
                "role": "default",
                "is_default": True,
                "projection": float(pa.project(default, axis, origin)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    desirability = pa.load_desirability(Path("configs/role_desirability.json"))
    descriptions = pa.load_role_descriptions(
        Path("external/assistant-axis/data/roles/role_list.json")
    )
    OUT_DIR.mkdir(exist_ok=True)

    projections = stage_projections(pa.load_vectors(DATA_DIR), pa.load_axes(DATA_DIR))
    panels = []
    for a, b in [("SFT", "DPO"), ("DPO", "RLVR")]:
        pairs = pa.pair_table(
            projections[projections["stage"].isin([a, b])], a, b, desirability, descriptions
        )
        rho = pairs[a].corr(pairs[b], method="spearman")
        panels.append(
            pc.projection_scatter(
                pairs, a, b, desirability, "projection", f"{a} vs {b} (Spearman {rho:.2f})"
            )
        )
    spec = (
        alt.hconcat(*panels, spacing=30)
        .resolve_legend(color="shared")
        .properties(
            title=alt.TitleParams(
                f"Layer {LAYER}: raw projections on each stage's assistant axis (hover for role)",
                anchor="middle",
                fontSize=15,
            )
        )
        .to_dict()
    )
    # One legend for both panels: drop it from the left panel's layers.
    for layer in spec["hconcat"][0]["layer"]:
        color = layer.get("encoding", {}).get("color")
        if color and "field" in color:
            color["legend"] = None
    alt.Chart.from_dict(spec).save(OUT_DIR / "projection_sft_dpo_rlvr.html")

    pass_rates = pa.load_pass_rates(DATA_DIR)
    pairs = pa.pair_table(
        pass_rates.assign(is_default=False),
        "SFT",
        "DPO",
        desirability,
        descriptions,
        value="pass_rate",
    )
    pc.projection_scatter(
        pairs,
        "SFT",
        "DPO",
        desirability,
        "tendency to role play",
        "Tendency to role play, SFT vs DPO (hover for role)",
    ).save(OUT_DIR / "role_play_tendency_sft_vs_dpo.html")


if __name__ == "__main__":
    main()
