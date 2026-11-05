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
    # Tendency to role play across SFT, DPO, RLVR

    Tendency to role play (`pass_rate` in the code): share of a role's judged responses scored 3 (fully in role) by judge `glm-5.3-flash`.
    Score 3 is the rule that selects the responses behind each persona vector.
    """)
    return


@app.cell
def load_pass_rates(mo, pa):
    DATA_DIR = mo.notebook_dir().parent / "data" / "phase-4"
    pass_rates = pa.load_pass_rates(DATA_DIR)
    pass_rates
    return (pass_rates,)


@app.cell
def load_labels(mo, pa):
    role_desirability = pa.load_desirability(
        mo.notebook_dir().parent / "configs" / "role_desirability.json"
    )
    role_descriptions = pa.load_role_descriptions(
        mo.notebook_dir().parent
        / "external"
        / "assistant-axis"
        / "data"
        / "roles"
        / "role_list.json"
    )
    return role_descriptions, role_desirability


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Across all stages
    """)
    return


@app.cell
def top_k_control(mo):
    top_k = mo.ui.number(start=1, stop=30, value=5, label="Top k per group")
    top_k
    return (top_k,)


@app.cell
def pass_rate_distribution(alt, np, pa, pass_rates, pd):
    # Letter-value (boxen) plot; Altair has no boxen mark, so each level is a bar spanning a pair of quantiles.
    # Level 1 is the middle half of the roles, each next level halves the tail share; floor(log2 n) - 3 levels keeps a few roles in the last tail.
    _n_levels = int(np.log2(pass_rates.groupby("stage").size().min())) - 3
    _rows = []
    for _stage, _rates in pass_rates.groupby("stage")["pass_rate"]:
        for _level in range(1, _n_levels + 1):
            _tail = 0.5 ** (_level + 1)
            _rows.append(
                {
                    "stage": _stage,
                    "level": _level,
                    "share": f"middle {1 - 2 * _tail:.1%}",
                    "low": _rates.quantile(_tail),
                    "high": _rates.quantile(1 - _tail),
                    "width": 70 * 0.75 ** (_level - 1),
                }
            )
    _levels = pd.DataFrame(_rows)
    _medians = pass_rates.groupby("stage", as_index=False)["pass_rate"].median()
    _x = alt.X("stage:N", sort=list(pa.STAGES), title=None, axis=alt.Axis(labelAngle=0))
    _boxes = (
        alt.Chart(_levels.sort_values("level", ascending=False))
        .mark_bar()
        .encode(
            x=_x,
            y=alt.Y("low:Q", scale=alt.Scale(domain=[0, 1]), title="tendency to role play"),
            y2="high:Q",
            size=alt.Size("width:Q", scale=None, legend=None),
            color=alt.Color("level:O", scale=alt.Scale(scheme="blues", reverse=True), legend=None),
            order=alt.Order("level:O", sort="descending"),
            tooltip=[
                "stage",
                "share",
                alt.Tooltip("low:Q", format=".2f"),
                alt.Tooltip("high:Q", format=".2f"),
            ],
        )
    )
    _median_ticks = (
        alt.Chart(_medians)
        .mark_tick(color="black", thickness=2, size=70)
        .encode(
            x=_x,
            y="pass_rate:Q",
            tooltip=["stage", alt.Tooltip("pass_rate:Q", format=".2f", title="median")],
        )
    )
    (_boxes + _median_ticks).properties(
        width=120 * len(pa.STAGES), height=420, title="Tendency to role play per stage"
    )
    return


@app.cell
def top_pass_rates(
    alt,
    pa,
    pass_rates,
    pc,
    role_descriptions,
    role_desirability,
    top_k,
):
    # The k lowest and k highest roles of each stage over all roles, colored by desirability.
    _roles = pass_rates.merge(role_desirability[["role", "label"]], on="role", how="left").fillna(
        {"label": "neutral"}
    )
    _roles = _roles.assign(description=_roles["role"].map(role_descriptions))
    alt.vconcat(
        *[
            pc.extremes_histogram(
                _roles[_roles["stage"] == _stage],
                "pass_rate",
                top_k.value,
                [0, 1],
                "tendency to role play",
                f"{_stage}: {top_k.value} roles least and most likely to role play",
            )
            for _stage in pa.STAGES
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stage pair
    """)
    return


@app.cell
def selectors(mo):
    pair_select = mo.ui.dropdown(
        options={
            "SFT vs DPO": ("SFT", "DPO"),
            "DPO vs RLVR": ("DPO", "RLVR"),
            "SFT vs RLVR": ("SFT", "RLVR"),
        },
        value="SFT vs DPO",
        label="Stages",
    )
    name_mode = mo.ui.radio(
        options={
            "random sample": "random",
            "top increasing": "increasing",
            "top decreasing": "decreasing",
        },
        value="random sample",
        label="Named roles",
        inline=True,
    )
    scatter_k = mo.ui.number(start=1, stop=30, value=5, label="k per group (top modes)")
    mo.vstack([pair_select, name_mode, scatter_k])
    return name_mode, pair_select, scatter_k


@app.cell
def pass_rate_pairs(
    pa,
    pair_select,
    pass_rates,
    role_descriptions,
    role_desirability,
):
    stage_a, stage_b = pair_select.value
    pass_rate_pairs = pa.pair_table(
        pass_rates.assign(is_default=False),
        stage_a,
        stage_b,
        role_desirability,
        role_descriptions,
        value="pass_rate",
    )
    pass_rate_pairs
    return pass_rate_pairs, stage_a, stage_b


@app.cell
def pass_rate_scatter(
    name_mode,
    pass_rate_pairs,
    pc,
    role_desirability,
    scatter_k,
    stage_a,
    stage_b,
):
    # Top modes name up to k roles per label group that actually moved that way; biggest moves first, so they keep their names when space runs out.
    if name_mode.value == "random":
        _named = None
    else:
        _decreasing = name_mode.value == "decreasing"
        _change = pass_rate_pairs["pass_rate_change"]
        _moved = pass_rate_pairs[_change < 0 if _decreasing else _change > 0]
        _ranked = _moved.sort_values("pass_rate_change", ascending=_decreasing)
        _named = _ranked[_ranked.groupby("label").cumcount() < scatter_k.value]["role"].tolist()
    pc.projection_scatter(
        pass_rate_pairs,
        stage_a,
        stage_b,
        role_desirability,
        "tendency to role play",
        f"Tendency to role play, {stage_a} vs {stage_b}",
        named_roles=_named,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Insufficient support

    A role gets no persona vector in a stage when it has fewer score-3 responses than the threshold; `4_vectors.py` uses `--min_count 50`.
    """)
    return


@app.cell
def support_control(mo):
    min_support = mo.ui.number(start=1, stop=1200, value=50, label="Minimum score-3 responses")
    min_support
    return (min_support,)


@app.cell
def insufficient_support(
    min_support,
    mo,
    pa,
    pass_rates,
    pd,
    role_desirability,
):
    _short = (
        pass_rates[pass_rates["n_pass"] < min_support.value]
        .merge(role_desirability[["role", "label"]], on="role", how="left")
        .fillna({"label": "neutral"})
    )
    _short = _short.assign(
        stage=pd.Categorical(_short["stage"], categories=list(pa.STAGES))
    ).sort_values(["stage", "n_pass"])
    mo.vstack(
        [
            mo.md(
                f"**{len(_short)} (stage, role) pairs below {min_support.value}**: "
                + ", ".join(f"{_stage} {(_short['stage'] == _stage).sum()}" for _stage in pa.STAGES)
            ),
            mo.ui.table(
                _short[["stage", "role", "label", "n_pass", "n_judged", "pass_rate"]].round(2),
                selection=None,
                show_column_summaries=False,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
