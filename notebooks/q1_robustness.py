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
    from persona_shift.robustness import permutation as perm
    from persona_shift.robustness import split_half as sh

    return alt, mo, np, pa, pd, perm, sh


@app.cell
def _(mo):
    mo.md(r"""
    # Q1 robustness: is the assistant-axis shift real?

    After DPO, desirable roles move toward the Assistant and undesirable roles don't.
    The checks below test whether that shift is real; symbols are defined in the next section.
    """)
    return


@app.cell(hide_code=True)
def headline_results(
    group_table,
    is_good,
    labelled,
    mo,
    shift,
    stage_a,
    stage_b,
    units,
):
    # Numbers come from the tables below; the wording was written for SFT → DPO in raw units.
    _t = group_table.set_index("version")
    _full, _matched, _gap = _t.loc["full"], _t.loc["matched"], _t.loc["gap"]
    _good_shifts = shift["full"][labelled][is_good]
    _bad_low = _full["bad shift"] - 1.96 * _full["bad se"]
    _bad_high = _full["bad shift"] + 1.96 * _full["bad se"]
    _lines = [
        "## Headline results",
        "",
        f"**Numbers:** {stage_a} → {stage_b}, layer 16, {units.value} units; they update with the selectors.",
        "",
        "**Holds:**",
        r"- **Desirable roles move toward the Assistant:** $\bar\Delta_D = "
        + f"{_full['good shift']:.3f}"
        + r"$, $z \approx "
        + f"{_full['good shift'] / _full['good se']:.1f}"
        + "$, far beyond sampling noise.",
        r"- **Undesirable roles barely move:** $\bar\Delta_U = "
        + f"{_full['bad shift']:.3f}"
        + r"$, 95% interval $["
        + f"{_bad_low:.2f}, {_bad_high:.2f}"
        + r"]$, well below $\bar\Delta_D$.",
        r"- **The contrast is real:** $\delta = "
        + f"{_full['contrast']:.3f}"
        + r"$, $z = "
        + f"{_full['contrast z']:.1f}"
        + "$.",
        r"- **Not from judge selection:** matching barely changes it ($\delta$: "
        + f"{_full['contrast']:.3f} → {_matched['contrast']:.3f}"
        + r"); $\delta_{\mathrm{gap}} = "
        + f"{_gap['contrast']:.3f}"
        + "$ (see Matched-sample comparison).",
        "",
        "**Caveats:**",
        r"- **The label explains little of the shift:** permutation $p = "
        + f"{_full['permutation p']:.3f}"
        + f"$, borderline; roles vary a lot within each group, and {(_good_shifts < 0).sum()} of {len(_good_shifts)} desirable roles move away from the Assistant (see Permutation test).",
        r"- **Selection moves all roles a little away from the Assistant:** gap $"
        + f"{_gap['good shift']:.3f}"
        + "$ for desirable and $"
        + f"{_gap['bad shift']:.3f}"
        + r"$ for undesirable roles, so it cancels in $\delta$.",
        "",
        "**Not yet checked:** z-score units, DPO → RLVR and SFT → RLVR, other layers.",
    ]
    _note = None
    if (stage_a, stage_b) != ("SFT", "DPO") or units.value != "raw":
        _note = mo.callout(
            mo.md(
                f"The wording was written for SFT → DPO in raw units; the current selection is {stage_a} → {stage_b}, {units.value}. Check it against the tables below."
            ),
            kind="warn",
        )
    mo.vstack([_x for _x in [mo.md("\n".join(_lines)), _note] if _x is not None])
    return


@app.cell(hide_code=True)
def definitions(mo):
    _cards = {
        "Inputs (fixed by the design)": r"""
    - **Stages** $s_1 \to s_2$ (e.g. SFT → DPO), layer 16.
    - **Roles** $R$: all roles in the role list.
      - $D \subset R$: desirable roles.
      - $U \subset R$: undesirable roles; the rest are neutral.
      - Labels from an LLM annotation of role names and descriptions.
    - **Items** $(p, q) \in I = P \times Q$: system prompt $p \in P$ ($|P| = 5$) and question $q \in Q$ ($|Q| = 240$).
    """,
        "Model (per stage)": r"""
    - **Activation** $a_s(r, p, q)$: mean activation over the response tokens, layer 16.
    """,
        "Judge": r"""
    - **Score** $c_s(r, p, q) \in \{0, 1, 2, 3\}$.
    - **Supported roles** $R_s \subset R$: roles with at least 50 responses scored 3 in stage $s$.
      - $R_{s_1, s_2} = R_{s_1} \cap R_{s_2}$: roles supported in both stages, the only ones with a shift.
    - **Kept items** per version, the items a role vector averages over:
      - $I_s^{\mathrm{full}}(r) = \{(p, q) \mid c_s(r, p, q) = 3\}$.
      - $I_s^{\mathrm{matched}}(r)$ for $r \in R_{s_1, s_2}$: $\{(p, q) \mid c_{s_1}(r, p, q) = 3 \text{ and } c_{s_2}(r, p, q) = 3\}$.
      - The default (the Assistant, no role prompt) keeps all of $I$ in both versions.
    - **Version values** $x_{\mathrm{full}}$, $x_{\mathrm{matched}}$: for $x \in \{\Delta(r), \bar\Delta_D, \bar\Delta_U, \delta\}$ (see Axis and shift), $x$ computed with role vectors built on $I_s^{\mathrm{full}}$ or $I_s^{\mathrm{matched}}$.
    - **gap:** $x_{\mathrm{gap}} = x_{\mathrm{full}} - x_{\mathrm{matched}}$.
    """,
        "Axis and shift (derived)": r"""
    - **Role vector** $v_s(r) = \frac{1}{|I_s(r)|} \sum_{(p, q) \in I_s(r)} a_s(r, p, q)$, with $I_s$ the version's kept items.
    - **Mean role vector** $\bar v_s = \frac{1}{|R_s|} \sum_{r \in R_s} v_s(r)$.
    - **Assistant axis** $u_s = \dfrac{v_s(\mathrm{default}) - \bar v_s}{\lVert v_s(\mathrm{default}) - \bar v_s \rVert}$.
    - **Projection** $\pi_s(r) = (v_s(r) - \bar v_s)^\top u_s$; the z-score unit standardizes $\pi_s(r)$ within each stage.
    - **Shift** $\Delta(r) = \pi_{s_2}(r) - \pi_{s_1}(r)$; positive means toward the Assistant.
    - **Group shift** $\bar\Delta_D$, $\bar\Delta_U$: mean shift over $D \cap R_{s_1, s_2}$ and $U \cap R_{s_1, s_2}$.
    - **Desirability contrast** $\delta = \bar\Delta_D - \bar\Delta_U$.
    """,
        "Noise": r"""
    - **Split** $k$: $Q = Q_k^{(a)} \cup Q_k^{(b)}$, half each; the same split for every role, both stages and both versions.
    - **Half values** $x_k^{(a)}$, $x_k^{(b)}$: $x$ with $I_s(r)$ replaced by $I_s(r) \cap (P \times Q_k^{(a)})$, or by $I_s(r) \cap (P \times Q_k^{(b)})$.
    - **Standard error** $\mathrm{se}(x) = \tfrac{1}{2}\,\mathrm{sd}_k\big(x_k^{(a)} - x_k^{(b)}\big)$.
    - **z** $= x / \mathrm{se}(x)$.
    - **Reliability** $\rho_k = \mathrm{corr}_{r \in R_s}\big(\pi_s(r)_k^{(a)},\ \pi_s(r)_k^{(b)}\big)$, per stage and version; reported as mean $\pm$ sd over $k$.
    """,
    }
    _card = {
        _title: mo.callout(mo.md(_text), kind="neutral", title=_title)
        for _title, _text in _cards.items()
    }
    mo.vstack(
        [
            mo.md("## Definitions"),
            mo.hstack(
                [_card["Inputs (fixed by the design)"], _card["Model (per stage)"], _card["Judge"]],
                widths=[3, 2, 4],
                align="stretch",
            ),
            mo.hstack(
                [_card["Axis and shift (derived)"], _card["Noise"]], widths="equal", align="stretch"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def setup_header(mo):
    mo.md("""
    ## Setup

    - Stage pair, units, rebuilt vectors.
    - Rebuilding takes ~2 min.
    """)
    return


@app.cell
def controls(mo):
    pair_select = mo.ui.dropdown(
        options={
            "SFT vs DPO": ("SFT", "DPO"),
            "DPO vs RLVR": ("DPO", "RLVR"),
            "SFT vs RLVR": ("SFT", "RLVR"),
        },
        value="SFT vs DPO",
        label="Stages",
    )
    units = mo.ui.radio(
        options={"raw projection": "raw", "z-score within stage": "z"},
        value="raw projection",
        label="Units",
        inline=True,
    )
    mo.vstack([pair_select, units])
    return pair_select, units


@app.cell
def load_scores(mo, pa):
    DATA_DIR = mo.notebook_dir().parent / "data" / "phase-4"
    LAYER = 16
    # 4_vectors.py --min_count 50: roles with fewer score-3 responses get no vector in that stage.
    MIN_COUNT = 50
    N_REPEATS = 100
    N_PERMUTATIONS = 10_000
    SEED = 0
    VERSIONS = ["full", "matched"]
    passed = {
        _stage: pa.judge_score_3_response_ids(pa.load_judge_scores(DATA_DIR, _stage))
        for _stage in pa.STAGES
    }
    role_desirability = pa.load_desirability(
        mo.notebook_dir().parent / "configs" / "role_desirability.json"
    )
    return (
        DATA_DIR,
        LAYER,
        MIN_COUNT,
        N_PERMUTATIONS,
        N_REPEATS,
        SEED,
        VERSIONS,
        passed,
        role_desirability,
    )


@app.cell
def run_checks(
    DATA_DIR,
    LAYER,
    MIN_COUNT,
    N_REPEATS,
    SEED,
    VERSIONS,
    mo,
    np,
    pa,
    pair_select,
    passed,
    sh,
):
    stage_a, stage_b = pair_select.value
    _supported = {
        _s: {_r for _r, _keys in passed[_s].items() if len(_keys) >= MIN_COUNT}
        for _s in (stage_a, stage_b)
    }
    common_roles = sorted(_supported[stage_a] & _supported[stage_b])
    _keeps = {
        ("full", _s): {_r: passed[_s][_r] for _r in _supported[_s]} for _s in (stage_a, stage_b)
    } | {
        ("matched", _s): {_r: passed[stage_a][_r] & passed[stage_b][_r] for _r in common_roles}
        for _s in (stage_a, stage_b)
    }

    def _run(version, stage):
        # A function, so the ~1 GB of per-question sums is freed before the next run.
        _roles, _sums, _counts = pa.sum_activations_by_question(
            DATA_DIR, stage, LAYER, _keeps[version, stage]
        )
        _vectors = _sums.sum(axis=1) / _counts.sum(axis=1)[:, None]
        _a, _b = sh.split_half(_sums, _counts, N_REPEATS, np.random.default_rng(SEED))
        return {
            "roles": _roles[:-1],
            "projection": pa.axis_projections(_vectors[:-1], _vectors[-1]),
            "half_a": _a,
            "half_b": _b,
            "n_items": _counts[:-1].sum(axis=1),
        }

    runs = {}
    for _version, _stage in mo.status.progress_bar(
        [(_v, _s) for _v in VERSIONS for _s in (stage_a, stage_b)],
        title="Rebuilding vectors and split halves",
    ):
        runs[_version, _stage] = _run(_version, _stage)
    return common_roles, runs, stage_a, stage_b


@app.cell
def shifts(VERSIONS, common_roles, pd, runs, stage_a, stage_b, units):
    def _standardize(values):
        # Rows are roles in the last axis; z-score each row within its stage.
        return (
            (values - values.mean(axis=-1, keepdims=True)) / values.std(axis=-1, keepdims=True)
            if units.value == "z"
            else values
        )

    def _stage(version, stage):
        _run = runs[version, stage]
        _projection = pd.Series(_standardize(_run["projection"]), index=_run["roles"])[common_roles]
        _differences = pd.DataFrame(
            _standardize(_run["half_a"]) - _standardize(_run["half_b"]), columns=_run["roles"]
        )[common_roles]
        return _projection, _differences

    # shift[version]: per role; noise[version]: its half-minus-half differences per repeat, same splits in both stages.
    shift, noise = {}, {}
    for _version in VERSIONS:
        (_proj_a, _diff_a), (_proj_b, _diff_b) = (
            _stage(_version, stage_a),
            _stage(_version, stage_b),
        )
        shift[_version], noise[_version] = _proj_b - _proj_a, _diff_b - _diff_a
    shift["gap"], noise["gap"] = shift["full"] - shift["matched"], noise["full"] - noise["matched"]
    _n_items = {
        _v: pd.Series(runs[_v, stage_a]["n_items"], index=runs[_v, stage_a]["roles"])[common_roles]
        for _v in VERSIONS
    }
    n_matched_items = _n_items["matched"]
    print(
        f"{len(common_roles)} roles in both stages; matched items per role: min {n_matched_items.min():.0f}, median {n_matched_items.median():.0f}"
    )
    return noise, shift


@app.cell
def group_summary(
    N_PERMUTATIONS,
    SEED,
    common_roles,
    noise,
    np,
    pd,
    perm,
    role_desirability,
    sh,
    shift,
):
    _labels = role_desirability.set_index("role")["label"].reindex(common_roles).fillna("neutral")
    labelled = _labels[_labels.isin(["desirable", "undesirable"])].index
    is_good = (_labels[labelled] == "desirable").to_numpy()

    def _with_se(value, per_repeat):
        return value, float(sh.full_data_se(np.asarray(per_repeat)[:, None])[0])

    _rows, permutation_nulls = [], {}
    for _version in ["full", "matched", "gap"]:
        _shift, _noise = shift[_version][labelled].to_numpy(), noise[_version][labelled].to_numpy()
        _observed, _null, _p = perm.permutation_test(
            _shift, is_good, N_PERMUTATIONS, np.random.default_rng(SEED)
        )
        permutation_nulls[_version] = (_observed, _null)
        _good, _good_se = _with_se(_shift[is_good].mean(), _noise[:, is_good].mean(axis=1))
        _bad, _bad_se = _with_se(_shift[~is_good].mean(), _noise[:, ~is_good].mean(axis=1))
        _contrast, _contrast_se = _with_se(
            _observed, [perm.mean_difference(_row, is_good) for _row in _noise]
        )
        _rows.append(
            {
                "version": _version,
                "good shift": _good,
                "good se": _good_se,
                "bad shift": _bad,
                "bad se": _bad_se,
                "contrast": _contrast,
                "contrast se": _contrast_se,
                "contrast z": _contrast / _contrast_se,
                "permutation p": _p,
            }
        )
    group_table = pd.DataFrame(_rows)
    print(f"{is_good.sum()} desirable and {(~is_good).sum()} undesirable roles")
    return group_table, is_good, labelled, permutation_nulls


@app.cell(hide_code=True)
def split_half_header(mo):
    mo.md(r"""
    ## Split-half test: is the shift larger than sampling noise?

    Main idea:

    - Measure the sampling noise (standard error) using split half, and compute the z-score of good/bad projection shift
    - The split half data is also used to compute the reliability

    **Table** (all score-3 responses are used, so called "full version"):

    - **rows:** good shift $\bar\Delta_D$, bad shift $\bar\Delta_U$, desirability contrast $\delta$.
    - **quantity:** its value, computed from all of $Q$ (no split).
    - **se:** its standard error from split-half, e.g. $\mathrm{se}(\delta)$ in the contrast row.
    - **z:** quantity / se, how many standard errors from 0.
    - **95% interval:** quantity $\pm 1.96\,$se.
    - **remark:** within noise for $|z| < 2$, beyond noise for $2 \le |z| < 3$, well beyond noise for $|z| \ge 3$; "small vs good shift" when the whole interval is closer to 0 than $\bar\Delta_D$.
    """)
    return


@app.cell
def split_half_table(group_table, mo):
    _full = group_table.set_index("version").loc["full"]
    _rows = [
        (r"good shift $\bar\Delta_D$", "good shift", "good se"),
        (r"bad shift $\bar\Delta_U$", "bad shift", "bad se"),
        (r"desirability contrast $\delta$", "contrast", "contrast se"),
    ]

    def _remark(value, se):
        # |z| = 2 is about the two-sided 5% level for a normal estimate.
        z = value / se
        if abs(z) >= 3:
            return "well beyond noise"
        if abs(z) >= 2:
            return "beyond noise"
        # Within noise only says "not distinguishable from 0"; "small" needs the whole interval below the good shift.
        if abs(value) + 1.96 * se < abs(_full["good shift"]):
            return "within noise, small vs good shift"
        return "within noise"

    _lines = ["| | quantity | se | z | 95% interval | remark |", "|---|--:|--:|--:|--:|---|"] + [
        f"| {_label} | {_full[_value]:.3f} | {_full[_se]:.3f} | {_full[_value] / _full[_se]:.2f} "
        f"| [{_full[_value] - 1.96 * _full[_se]:.3f}, {_full[_value] + 1.96 * _full[_se]:.3f}] "
        f"| {_remark(_full[_value], _full[_se])} |"
        for _label, _value, _se in _rows
    ]
    mo.md("\n".join(_lines))
    return


@app.cell
def reliability_table(VERSIONS, pd, runs, sh, stage_a, stage_b):
    # Split-half reliability: correlation over roles between the two halves' projections, per split.
    # The sd describes variation across splits; the splits share the data, so it is not a standard error.
    _rows = []
    for _v in VERSIONS:
        for _s in (stage_a, stage_b):
            _rho = pd.Series(sh.reliability(runs[_v, _s]["half_a"], runs[_v, _s]["half_b"]))
            _rows.append(
                {
                    "version": _v,
                    "stage": _s,
                    "reliability (mean ± sd over splits)": f"{_rho.mean():.3f} ± {_rho.std():.3f}",
                }
            )
    pd.DataFrame(_rows)
    return


@app.cell(hide_code=True)
def permutation_header(mo):
    mo.md(r"""
    ## Permutation test: does the desirability contrast beat a random grouping?

    - **Gray:** $\delta$ under shuffled good/bad labels.
    - **Red line:** observed $\delta$.
    - **remark:** compares the observed $|\delta|$ with the $|\delta|$ of the 10,000 shuffles.
      - $p < 0.05$: the observed $|\delta|$ is larger than at least 95% of the shuffled values.
      - $0.05 \le p < 0.1$: it is larger than 90 to 95% of them, which is borderline.
      - $p \ge 0.1$: it is larger than fewer than 90% of them, which is typical of random labels.
    """)
    return


@app.cell
def permutation_table(group_table, mo):
    def _remark(p):
        if p < 0.05:
            return r"$\lvert\delta\rvert$ above at least 95% of shuffles"
        if p < 0.1:
            return r"$\lvert\delta\rvert$ above 90 to 95% of shuffles (borderline)"
        return r"$\lvert\delta\rvert$ typical of shuffles"

    _table = group_table.set_index("version")
    _lines = [
        r"| | desirability contrast $\delta$ | permutation p | remark |",
        "|---|--:|--:|---|",
    ] + [
        f"| {_version} | {_row['contrast']:.3f} | {_row['permutation p']:.3f} | {_remark(_row['permutation p'])} |"
        for _version, _row in _table.iterrows()
    ]
    mo.md("\n".join(_lines))
    return


@app.cell
def permutation_chart(alt, np, pd, permutation_nulls):
    # Null distribution of the good-minus-bad contrast under shuffled labels; the red line is the observed contrast.
    # Binned in numpy because Altair refuses to embed more than 5000 rows.
    _charts = []
    for _version, (_observed, _null) in permutation_nulls.items():
        _counts, _edges = np.histogram(_null, bins=60)
        _bins = pd.DataFrame({"low": _edges[:-1], "high": _edges[1:], "count": _counts})
        _hist = (
            alt.Chart(_bins)
            .mark_bar(color="lightgray")
            .encode(
                x=alt.X("low:Q", title="desirability contrast"),
                x2="high:Q",
                y=alt.Y("count:Q", title=None),
            )
        )
        _line = (
            alt.Chart(pd.DataFrame({"contrast": [_observed]}))
            .mark_rule(color="firebrick", strokeWidth=2)
            .encode(x="contrast:Q")
        )
        _charts.append((_hist + _line).properties(width=260, height=160, title=_version))
    alt.hconcat(*_charts)
    return


@app.cell(hide_code=True)
def selection_header(mo):
    mo.md(r"""
    ## Matched-sample comparison (is the shift in the model or in which responses pass?)

    - **gap:** full minus matched, the part of $\delta$ due to selection.
    - **contrast z:** $\delta / \mathrm{se}(\delta)$ per version.
    """)
    return


@app.cell
def matched_table(group_table):
    group_table.set_index("version")[["contrast", "contrast se", "contrast z"]].round(3)
    return


@app.cell(hide_code=True)
def summary_header(mo):
    mo.md(r"""
    ## Summary

    - Every check per version, in one table.
    - **good / bad shift:** $\bar\Delta_D$, $\bar\Delta_U$; **contrast:** desirability contrast $\delta$.
    """)
    return


@app.cell
def summary_table(group_table):
    group_table.round(3)
    return


if __name__ == "__main__":
    app.run()
