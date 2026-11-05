import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from persona_shift import persona_axes as pa
    from persona_shift import persona_charts as pc

    return alt, mo, np, pa, pc, pd


@app.cell
def _(mo):
    mo.md("""
    # Assistant axis across SFT, DPO, RLVR

    Axis per stage, after Lu et al.: mean `default` vector minus the mean of all role vectors.
    Data: `data/phase-4/{sft,dpo,rlvr}-sysprompt`, judge `glm-5.3-flash`.
    """)
    return


@app.cell
def selectors(LAYERS, mo):
    layer_select = mo.ui.dropdown(options={str(l): l for l in LAYERS}, value="16", label="Layer")
    pair_select = mo.ui.dropdown(
        options={
            "SFT vs DPO": ("SFT", "DPO"),
            "DPO vs RLVR": ("DPO", "RLVR"),
            "SFT vs RLVR": ("SFT", "RLVR"),
        },
        value="SFT vs DPO",
        label="Stages",
    )
    mo.hstack([layer_select, pair_select], justify="start")
    return layer_select, pair_select


@app.cell
def load_vectors(mo, pa):
    DATA_DIR = mo.notebook_dir().parent / "data" / "phase-4"
    STAGES = pa.STAGES
    LAYERS = pa.LAYERS
    vectors = pa.load_vectors(DATA_DIR)
    print(vectors.groupby(["stage", "layer", "is_default"]).size())
    vectors
    return DATA_DIR, LAYERS, STAGES, vectors


@app.cell
def load_axes(DATA_DIR, pa):
    axes = pa.load_axes(DATA_DIR)
    axes
    return (axes,)


@app.cell
def check_axes(axes, np, pd, vectors):
    # 5_axis.py subtracts in bf16, the dtype the vectors are stored in, so compare against a float32 recompute by direction and relative size.
    _recomputed = {
        _key: _group.loc[_group["is_default"], "vector"].iloc[0]
        - np.stack(_group.loc[~_group["is_default"], "vector"]).mean(axis=0)
        for _key, _group in vectors.groupby(["stage", "layer"])
    }
    _rows = []
    for _r in axes.itertuples():
        _f32 = _recomputed[(_r.stage, _r.layer)]
        _rows.append(
            {
                "stage": _r.stage,
                "layer": _r.layer,
                "axis_norm": float(np.linalg.norm(_r.axis)),
                "cos_vs_float32": float(
                    _f32 @ _r.axis / (np.linalg.norm(_f32) * np.linalg.norm(_r.axis))
                ),
                "rel_diff_vs_float32": float(np.linalg.norm(_f32 - _r.axis) / np.linalg.norm(_f32)),
            }
        )
    axis_check = pd.DataFrame(_rows)
    assert (axis_check["cos_vs_float32"] > 0.999).all(), axis_check
    axis_check
    return


@app.cell
def load_desirability(mo, pa, vectors):
    # First-pass labels by an LLM agent from role names and descriptions only: would people want an AI assistant to adopt this persona?
    role_desirability = pa.load_desirability(
        mo.notebook_dir().parent / "configs" / "role_desirability.json"
    )
    _ours = role_desirability[role_desirability["role"].isin(vectors["role"])]
    mo.vstack(
        [
            mo.md(
                f"**Desirability labels**: {len(role_desirability)} roles, {len(_ours)} in this run"
            ),
            _ours.groupby("label").agg(roles=("role", "size"), borderline=("borderline", "sum")),
        ]
    )
    return (role_desirability,)


@app.cell
def load_role_descriptions(mo, pa):
    # One-line role descriptions from the assistant-axis role list, shown in plot tooltips.
    role_descriptions = pa.load_role_descriptions(
        mo.notebook_dir().parent
        / "external"
        / "assistant-axis"
        / "data"
        / "roles"
        / "role_list.json"
    )
    return (role_descriptions,)


@app.cell
def selection(axes, layer_select, mo, pair_select, vectors):
    stage_a, stage_b = pair_select.value
    _keep = lambda df: df[
        (df["layer"] == layer_select.value) & df["stage"].isin([stage_a, stage_b])
    ]
    selected_vectors = _keep(vectors)
    selected_axes = _keep(axes)

    mo.vstack(
        [
            mo.md(
                f"**Layer {layer_select.value}, {stage_a} vs {stage_b}**: {len(selected_vectors)} vectors, {len(selected_axes)} axes"
            ),
            selected_vectors.groupby(["stage", "is_default"]).size(),
            selected_axes,
        ]
    )
    return selected_axes, stage_a, stage_b


@app.cell
def projections(
    layer_select,
    pa,
    pd,
    selected_axes,
    stage_a,
    stage_b,
    vectors,
):
    # Each stage's personas on its own axis, relative to that stage's mean role vector, so 0 is the average persona.
    _rows = []
    for _stage in [stage_a, stage_b]:
        _roles, _matrix, _default = pa.stage_matrix(vectors, _stage, layer_select.value)
        _axis = selected_axes.loc[selected_axes["stage"] == _stage, "axis"].iloc[0]
        _origin = _matrix.mean(axis=0)
        _rows += [
            {"stage": _stage, "role": _role, "is_default": False, "projection": float(_p)}
            for _role, _p in zip(_roles, pa.project(_matrix, _axis, _origin))
        ]
        _rows.append(
            {
                "stage": _stage,
                "role": "default",
                "is_default": True,
                "projection": float(pa.project(_default, _axis, _origin)),
            }
        )
    projections = pd.DataFrame(_rows)
    projections
    return (projections,)


@app.cell
def axis_histograms(
    layer_select,
    pc,
    projections,
    role_descriptions,
    role_desirability,
    stage_a,
    stage_b,
):
    pc.histogram_pair(
        projections,
        stage_a,
        stage_b,
        role_desirability,
        role_descriptions,
        "projection on the stage's assistant axis",
        f"layer {layer_select.value}",
    )
    return


@app.cell
def projection_scatter(
    layer_select,
    pa,
    pc,
    projections,
    role_descriptions,
    role_desirability,
    stage_a,
    stage_b,
):
    projection_pairs = pa.pair_table(
        projections, stage_a, stage_b, role_desirability, role_descriptions
    )
    _spearman = projection_pairs[stage_a].corr(projection_pairs[stage_b], method="spearman")
    pc.projection_scatter(
        projection_pairs,
        stage_a,
        stage_b,
        role_desirability,
        "projection",
        f"Layer {layer_select.value}: raw projections, {stage_a} vs {stage_b} (Spearman {_spearman:.2f})",
    )
    return (projection_pairs,)


@app.cell
def stage_positions(STAGES, axes, layer_select, pa, pd, vectors):
    # All three stages at the selected layer: the raw projection, and its z-score within each stage so the overall widening between stages doesn't read as every persona moving.
    _rows = []
    for _stage in STAGES:
        _roles, _matrix, _ = pa.stage_matrix(vectors, _stage, layer_select.value)
        _axis = axes.loc[
            (axes["stage"] == _stage) & (axes["layer"] == layer_select.value), "axis"
        ].iloc[0]
        _projection = pa.project(_matrix, _axis, _matrix.mean(axis=0))
        _rows += [
            {"stage": _stage, "role": _role, "projection": float(_p), "z": float(_z)}
            for _role, _p, _z in zip(_roles, _projection, pa.zscore(_projection))
        ]
    stage_positions = pd.DataFrame(_rows)
    stage_positions
    return (stage_positions,)


@app.cell
def highlight_selector(mo):
    highlight_mode = mo.ui.radio(
        options=["top movers", "desirability labels"],
        value="top movers",
        label="Highlight",
        inline=True,
    )
    n_highlight = mo.ui.slider(1, 15, value=5, show_value=True, label="Highlighted roles per group")
    mo.hstack([highlight_mode, n_highlight], justify="start", gap=2)
    return highlight_mode, n_highlight


@app.cell
def slope_chart(
    STAGES,
    alt,
    highlight_mode,
    layer_select,
    n_highlight,
    pc,
    role_descriptions,
    role_desirability,
    stage_a,
    stage_b,
    stage_positions,
):
    if highlight_mode.value == "top movers":
        _groups = pc.top_mover_groups(stage_positions, stage_a, stage_b, n_highlight.value)
        _scale = alt.Scale(domain=["rose", "dropped"], range=["steelblue", "firebrick"])
        _legend_title, _what = (
            f"{stage_a} → {stage_b}",
            f"top {n_highlight.value} risers and droppers {stage_a} → {stage_b}",
        )
    else:
        _groups = pc.top_mover_labels(
            stage_positions, stage_a, stage_b, role_desirability, n_highlight.value
        )
        _scale = pc.LABEL_SCALE
        _legend_title, _what = (
            "persona label",
            f"top {n_highlight.value} movers among desirable and among undesirable personas",
        )
    pc.slope_chart(
        stage_positions,
        list(STAGES),
        stage_a,
        stage_b,
        _groups,
        _scale,
        _legend_title,
        role_descriptions,
        "projection on the stage's assistant axis",
        f"Layer {layer_select.value}: persona positions across stages, {_what}",
        headings=highlight_mode.value != "top movers",
    )
    return


@app.cell
def k_selector(mo):
    k_select = mo.ui.slider(1, 50, value=10, show_value=True, label="Personas per table")
    return (k_select,)


@app.cell
def change_tables(
    k_select,
    mo,
    pc,
    projection_pairs,
    role_desirability,
    stage_a,
    stage_b,
):
    rank_changes = projection_pairs[["role", f"{stage_a}_rank", f"{stage_b}_rank"]].assign(
        rank_change=projection_pairs[f"{stage_b}_rank"] - projection_pairs[f"{stage_a}_rank"]
    )
    # Raw change on each stage's own scale, so it includes any overall widening of the projection spread between the stages.
    projection_changes = projection_pairs[["role", stage_a, stage_b]].assign(
        projection_change=projection_pairs[stage_b] - projection_pairs[stage_a]
    )

    def _side_by_side(table, column, what):
        return mo.hstack(
            [
                mo.vstack(
                    [
                        mo.md(f"**{what} rose most**, {stage_a} → {stage_b}"),
                        pc.style_by_label(
                            table.nlargest(k_select.value, column), role_desirability
                        ),
                    ]
                ),
                mo.vstack(
                    [
                        mo.md(f"**{what} dropped most**, {stage_a} → {stage_b}"),
                        pc.style_by_label(
                            table.nsmallest(k_select.value, column), role_desirability
                        ),
                    ]
                ),
            ],
            justify="start",
            gap=2,
        )

    mo.accordion(
        {
            "Details: top movers by rank and by projection change": mo.vstack(
                [
                    k_select,
                    _side_by_side(rank_changes, "rank_change", "Rank"),
                    _side_by_side(projection_changes, "projection_change", "Projection"),
                ]
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
