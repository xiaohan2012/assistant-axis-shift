"""Altair charts of personas placed along one direction in two or more stages.

Every chart takes a positions table with columns stage, role and projection (plus is_default where the Assistant is shown).
"""

import altair as alt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from scipy.stats import gaussian_kde

LABEL_COLORS = {"desirable": "forestgreen", "undesirable": "purple"}
LABEL_SCALE = alt.Scale(domain=list(LABEL_COLORS), range=list(LABEL_COLORS.values()))
DIRECTION_DASH = alt.Scale(domain=["rising", "dropping"], range=[[1, 0], [6, 4]])


def spread_labels(values: list[float], gap: float) -> list[float]:
    """Positions for sorted label values, at least gap apart, each crowded block centered on its values' mean."""
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value])
        while len(blocks) > 1:
            lower, upper = blocks[-2], blocks[-1]
            if (
                np.mean(upper)
                - (len(upper) - 1) * gap / 2
                - (np.mean(lower) + (len(lower) - 1) * gap / 2)
                >= gap
            ):
                break
            blocks[-2:] = [lower + upper]
    return [
        float(np.mean(block) + (i - (len(block) - 1) / 2) * gap)
        for block in blocks
        for i in range(len(block))
    ]


def sample_named_roles(
    desirability: pd.DataFrame, name_fraction: float = 0.4, seed: int = 0
) -> set[str]:
    """A seeded random name_fraction of the desirable and of the undesirable roles, to name in crowded plots.

    Drawn from role names, not positions, so every stage and every chart names the same roles.
    """
    rng = np.random.default_rng(seed)
    named: set[str] = set()
    for label in LABEL_COLORS:
        group_roles = sorted(desirability.loc[desirability["label"] == label, "role"])
        named |= set(
            rng.choice(group_roles, size=round(name_fraction * len(group_roles)), replace=False)
        )
    return named


def place_names(
    xs: list[float],
    ys: list[float],
    names: list[str],
    x_domain: list[float],
    y_domain: list[float],
    width_px: int,
    height_px: int,
    char_px: float = 6.5,
    line_px: float = 12.0,
    obstacles: list[tuple[float, float]] | None = None,
    obstacle_px: float = 4.0,
) -> list[tuple[int, float, float]]:
    """Greedy label placement: each name goes beside its point, wherever it overlaps no earlier name and no obstacle point.

    Candidate spots are right, left, above, below, then the diagonals, then the same a step further out.
    Names are tried in the given order; one with no free spot is dropped.
    Returns (index, left edge x, center y) in data units for each placed name.
    """
    x_scale = (x_domain[1] - x_domain[0]) / width_px
    y_scale = (y_domain[1] - y_domain[0]) / height_px
    placed: list[tuple[float, float, float, float]] = []
    for ox, oy in obstacles or []:
        cx, cy = (ox - x_domain[0]) / x_scale, (oy - y_domain[0]) / y_scale
        placed.append((cx - obstacle_px, cx + obstacle_px, cy - obstacle_px, cy + obstacle_px))
    result = []
    for i, (x, y, name) in enumerate(zip(xs, ys, names)):
        px, py = (x - x_domain[0]) / x_scale, (y - y_domain[0]) / y_scale
        width = len(name) * char_px
        # (left edge, center y) of the name's box for each candidate spot, in pixels with y up.
        spots = []
        for step in (1, 2):
            gap, rise = 6 * step, line_px * step
            spots += [
                (px + gap, py),
                (px - gap - width, py),
                (px - width / 2, py + rise),
                (px - width / 2, py - rise),
                (px + gap, py + rise),
                (px - gap - width, py + rise),
                (px + gap, py - rise),
                (px - gap - width, py - rise),
            ]
        for left, center in spots:
            box = (left, left + width, center - line_px / 2, center + line_px / 2)
            inside = box[0] >= 0 and box[1] <= width_px and box[2] >= 0 and box[3] <= height_px
            if inside and not any(
                box[0] < b[1] and b[0] < box[1] and box[2] < b[3] and b[2] < box[3] for b in placed
            ):
                placed.append(box)
                result.append((i, x_domain[0] + left * x_scale, y_domain[0] + center * y_scale))
                break
    return result


def histogram_bins(domain: list[float], max_bins: int = 30) -> tuple[alt.Bin, np.ndarray]:
    """Bins of a round width over domain, as an Altair bin and as the matching numpy edges."""
    step = next(
        step
        for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0)
        if (domain[1] - domain[0]) / step <= max_bins
    )
    extent = [np.floor(domain[0] / step) * step, np.ceil(domain[1] / step) * step]
    return alt.Bin(step=step, extent=extent), np.arange(extent[0], extent[1] + step / 2, step)


def density_curve(values: np.ndarray, grid: np.ndarray, bin_step: float) -> np.ndarray:
    """Gaussian kernel density of values on grid (Scott's bandwidth), scaled to expected counts per bin of width bin_step."""
    return gaussian_kde(values)(grid) * len(values) * bin_step


def axis_histogram(
    positions: pd.DataFrame,
    stage: str,
    domain: list[float],
    desirability: pd.DataFrame,
    descriptions: pd.Series,
    axis_title: str,
    title: str,
    names_per_row: int = 5,
    row_px: float = 14.0,
    font_px: float = 11.0,
    height_px: int = 420,
    name_fraction: float = 0.4,
    seed: int = 0,
) -> alt.LayerChart:
    """Histogram of one stage's personas along a direction.

    All desirable and undesirable roles are marked; those from sample_named_roles are named below the axis in rows row_px apart, desirable and undesirable rows alternating.
    The default Assistant is a dashed line when present.
    """
    stage_rows = positions[positions["stage"] == stage]
    roles = stage_rows[~stage_rows["is_default"]]
    roles = roles.assign(description=roles["role"].map(descriptions))
    default = stage_rows[stage_rows["is_default"]]
    # Explicit bins shared with Vega-Lite, so the tallest bar computed here is the one drawn and the pinned y range fits it.
    bins, edges = histogram_bins(domain)
    top = np.histogram(roles["projection"], bins=edges)[0].max()

    tagged = roles.merge(desirability[["role", "label"]], on="role")
    tagged = tagged[tagged["label"] != "neutral"]
    # Rows take every n-th role by position, so neighbours on the axis land in different rows; each row is then spread along x.
    # Space names in a row by the row's widest name at this font size (about 0.6 em per character), plus a margin, in data units.
    data_per_px = (domain[1] - domain[0]) / 640
    sampled = sample_named_roles(desirability, name_fraction, seed)
    groups = {
        label: tagged[(tagged["label"] == label) & tagged["role"].isin(sampled)].sort_values(
            "projection"
        )
        for label in LABEL_COLORS
    }
    n_rows = {
        label: max(1, int(np.ceil(len(group) / names_per_row))) for label, group in groups.items()
    }
    # Desirable and undesirable rows alternate; the group with more rows continues alone at the bottom.
    row_order = [
        (label, row)
        for row in range(max(n_rows.values()))
        for label in LABEL_COLORS
        if row < n_rows[label]
    ]
    # Rows are a fixed row_px apart whatever the bar height: solve for the data step that the y scale, from the lowest row to the bars' top, maps to row_px.
    total_rows = len(row_order)
    chart_top = top * 1.15
    row_step = row_px * chart_top / (height_px - row_px * (total_rows + 1))
    rows = []
    for row_index, (label, row) in enumerate(row_order):
        part = groups[label].iloc[row :: n_rows[label]].copy()
        widest_px = part["role"].str.len().max() * 0.6 * font_px + 10
        part["label_x"] = spread_labels(part["projection"].tolist(), widest_px * data_per_px)
        part["y"] = -row_step * (1 + row_index)
        rows.append(part)
    named = pd.concat(rows)
    # Connectors stop short of the marker (6 px) and of the name (half a line plus 3 px), so they don't touch either.
    data_per_px_y = row_step / row_px
    named["stem_start"] = -6 * data_per_px_y
    named["stem_end"] = named["y"] + (font_px / 2 + 3) * data_per_px_y

    x = alt.X("projection:Q", scale=alt.Scale(domain=domain), title=axis_title)
    position_color = alt.Color(
        "projection:Q", scale=alt.Scale(scheme="redblue", domainMid=0), legend=None
    )
    label_color = alt.Color("label:N", scale=LABEL_SCALE, title="persona label")
    label_color_no_legend = alt.Color("label:N", scale=LABEL_SCALE, legend=None)
    bars = (
        alt.Chart(roles)
        .mark_bar(opacity=0.45)
        .encode(
            x=alt.X(
                "projection:Q",
                bin=bins,
                scale=alt.Scale(domain=domain),
                title=axis_title,
            ),
            y=alt.Y(
                "count()",
                title="personas",
                scale=alt.Scale(domain=[-row_step * (total_rows + 1), chart_top], nice=False),
                axis=alt.Axis(labelExpr="datum.value < 0 ? '' : datum.label"),
            ),
            color=alt.Color(
                "projection:Q",
                bin=bins,
                scale=alt.Scale(scheme="redblue", domainMid=0),
                legend=None,
            ),
        )
    )
    # Each group's smoothed distribution on the histogram's count scale, and its mean as a dashed rule up to the curve.
    grid = np.linspace(domain[0], domain[1], 200)
    bin_step = edges[1] - edges[0]
    curves, means = [], []
    for label in LABEL_COLORS:
        values = tagged.loc[tagged["label"] == label, "projection"].to_numpy()
        curve = density_curve(values, grid, bin_step)
        curves.append(pd.DataFrame({"projection": grid, "count": curve, "label": label}))
        means.append(
            {
                "label": label,
                "projection": values.mean(),
                "height": float(np.interp(values.mean(), grid, curve)),
            }
        )
    group_curves = (
        alt.Chart(pd.concat(curves))
        .mark_line(strokeWidth=2.5, opacity=0.9)
        .encode(x=x, y="count:Q", color=label_color_no_legend, detail="label:N")
    )
    group_means = (
        alt.Chart(pd.DataFrame(means))
        .mark_rule(strokeDash=[5, 3], strokeWidth=1.5)
        .encode(
            x=x,
            y="height:Q",
            y2=alt.datum(0),
            color=label_color_no_legend,
            tooltip=["label", alt.Tooltip("projection:Q", title="mean projection", format=".2f")],
        )
    )
    rug = (
        alt.Chart(roles)
        .mark_point(shape="diamond", filled=True, size=60, opacity=0.8)
        .encode(
            x=x,
            y=alt.datum(0),
            color=position_color,
            tooltip=["role", "description", alt.Tooltip("projection:Q", format=".2f")],
        )
    )
    # Connectors stay faint unless their name is hovered; the selection is named per stage because the two stages' charts share one spec.
    hover = alt.selection_point(
        name=f"hover_{stage}", fields=["role"], on="pointerover", clear="pointerout", empty=False
    )
    tagged_stems = (
        alt.Chart(named)
        .mark_rule()
        .encode(
            x=alt.X("label_x:Q", scale=alt.Scale(domain=domain), title=axis_title),
            y="stem_end:Q",
            x2="projection:Q",
            y2="stem_start:Q",
            color=label_color_no_legend,
            opacity=alt.condition(hover, alt.value(0.55), alt.value(0.08)),
            strokeWidth=alt.condition(hover, alt.value(1.5), alt.value(1.0)),
        )
    )
    tagged_marks = (
        alt.Chart(tagged)
        .mark_point(shape="triangle-up", filled=True, size=90)
        .encode(
            x=x,
            y=alt.datum(0),
            color=label_color,
            tooltip=["role", "label", "description", alt.Tooltip("projection:Q", format=".2f")],
        )
    )
    tagged_names = (
        alt.Chart(named)
        .mark_text(baseline="middle", fontSize=font_px, fontWeight="bold")
        .encode(x="label_x:Q", y="y:Q", text="role", color=label_color_no_legend)
        .add_params(hover)
    )
    assistant = alt.Chart(default).mark_rule(color="black", strokeDash=[4, 3]).encode(x=x)
    assistant_name = (
        alt.Chart(default)
        .mark_text(text="default (Assistant)", dy=-6, align="right")
        .encode(x=x, y=alt.datum(top * 1.05))
    )
    return (
        bars
        + group_curves
        + group_means
        + rug
        + tagged_stems
        + tagged_marks
        + tagged_names
        + assistant
        + assistant_name
    ).properties(title=title, width=640, height=height_px)


def extremes_histogram(
    roles: pd.DataFrame,
    value: str,
    k: int,
    domain: list[float],
    axis_title: str,
    title: str,
    names_per_row: int = 3,
    row_px: float = 14.0,
    font_px: float = 11.0,
    height_px: int = 300,
) -> alt.LayerChart:
    """Histogram of roles by `value`, with the k lowest and k highest roles named below the axis, colored by label.

    `roles` has columns role, label (desirable, undesirable, neutral), description and `value`.
    """
    bins, edges = histogram_bins(domain)
    top = np.histogram(roles[value], bins=edges)[0].max()
    ranked = roles.sort_values(value)
    named = pd.concat([ranked.head(k), ranked.tail(k)]).drop_duplicates("role")

    # Same row layout as axis_histogram: every n-th name per row, rows a fixed row_px apart, names in a row spaced by its widest name.
    n_rows = int(np.ceil(len(named) / names_per_row))
    chart_top = top * 1.15
    row_step = row_px * chart_top / (height_px - row_px * (n_rows + 1))
    data_per_px = (domain[1] - domain[0]) / 640
    rows = []
    for row in range(n_rows):
        part = named.iloc[row::n_rows].copy()
        widest_px = part["role"].str.len().max() * 0.6 * font_px + 10
        gap = widest_px * data_per_px
        # The named roles sit at both ends of the axis, so keep each name inside the chart, then restore the gaps from each side.
        spread = np.clip(
            spread_labels(part[value].tolist(), gap), domain[0] + gap / 2, domain[1] - gap / 2
        )
        for i in range(1, len(spread)):
            spread[i] = max(spread[i], spread[i - 1] + gap)
        for i in range(len(spread) - 2, -1, -1):
            spread[i] = min(spread[i], spread[i + 1] - gap)
        part["label_x"] = spread
        part["y"] = -row_step * (1 + row)
        rows.append(part)
    named = pd.concat(rows)
    data_per_px_y = row_step / row_px
    named["stem_start"] = -6 * data_per_px_y
    named["stem_end"] = named["y"] + (font_px / 2 + 3) * data_per_px_y

    scale = alt.Scale(domain=domain)
    x = alt.X(f"{value}:Q", scale=scale, title=axis_title)
    label_scale = alt.Scale(
        domain=[*LABEL_COLORS, "neutral"], range=[*LABEL_COLORS.values(), "dimgray"]
    )
    label_color = alt.Color("label:N", scale=label_scale, title="persona label")
    tooltip = ["role", "label", "description", alt.Tooltip(f"{value}:Q", format=".2f")]
    bars = (
        alt.Chart(roles)
        .mark_bar(color="gray", opacity=0.35)
        .encode(
            x=alt.X(f"{value}:Q", bin=bins, scale=scale, title=axis_title),
            y=alt.Y(
                "count()",
                title="personas",
                scale=alt.Scale(domain=[-row_step * (n_rows + 1), chart_top], nice=False),
                axis=alt.Axis(labelExpr="datum.value < 0 ? '' : datum.label"),
            ),
        )
    )
    stems = (
        alt.Chart(named)
        .mark_rule(opacity=0.4)
        .encode(
            x=alt.X("label_x:Q", scale=scale, title=axis_title),
            y="stem_end:Q",
            x2=f"{value}:Q",
            y2="stem_start:Q",
            color=alt.Color("label:N", scale=label_scale, legend=None),
        )
    )
    marks = (
        alt.Chart(named)
        .mark_point(shape="triangle-up", filled=True, size=90)
        .encode(x=x, y=alt.datum(0), color=label_color, tooltip=tooltip)
    )
    names = (
        alt.Chart(named)
        .mark_text(baseline="middle", fontSize=font_px, fontWeight="bold")
        .encode(
            x="label_x:Q",
            y="y:Q",
            text="role",
            color=alt.Color("label:N", scale=label_scale, legend=None),
            tooltip=tooltip,
        )
    )
    return (bars + stems + marks + names).properties(title=title, width=640, height=height_px)


def histogram_pair(
    positions: pd.DataFrame,
    stage_a: str,
    stage_b: str,
    desirability: pd.DataFrame,
    descriptions: pd.Series,
    axis_title: str,
    title_suffix: str,
) -> alt.VConcatChart:
    """axis_histogram for two stages, stacked, on a shared x range."""
    low, high = positions["projection"].min(), positions["projection"].max()
    pad = 0.05 * (high - low)
    domain = [low - pad, high + pad]
    return alt.vconcat(
        *[
            axis_histogram(
                positions,
                stage,
                domain,
                desirability,
                descriptions,
                axis_title,
                f"{stage}, {title_suffix}",
            )
            for stage in [stage_a, stage_b]
        ]
    )


def projection_scatter(
    pairs: pd.DataFrame,
    stage_a: str,
    stage_b: str,
    desirability: pd.DataFrame,
    axis_title: str,
    title: str,
    name_fraction: float = 0.4,
    seed: int = 0,
    named_roles: list[str] | None = None,
) -> alt.LayerChart:
    """Each persona's projection in stage_a against stage_b, colored by label, with the diagonal.

    `pairs` is a persona_axes.pair_table.
    Roles in `named_roles`, in priority order, are named where place_names finds room; without it, a sample_named_roles sample of desirable and undesirable roles is.
    """
    low = pairs[[stage_a, stage_b]].min().min() - 0.1
    high = pairs[[stage_a, stage_b]].max().max() + 0.1
    scale = alt.Scale(domain=[low, high])
    x = alt.X(f"{stage_a}:Q", scale=scale, title=f"{stage_a} {axis_title}")
    y = alt.Y(f"{stage_b}:Q", scale=scale, title=f"{stage_b} {axis_title}")
    tooltip = [
        "role",
        "label",
        "description",
        alt.Tooltip(f"{stage_a}:Q", format=".2f"),
        alt.Tooltip(f"{stage_b}:Q", format=".2f"),
        f"{stage_a}_rank:Q",
        f"{stage_b}_rank:Q",
    ]
    tagged = pairs[pairs["label"] != "neutral"]
    label_color = alt.Color("label:N", scale=LABEL_SCALE, title="persona label")
    # Neutral personas as a gray backdrop, desirable and undesirable ones in the histogram's colors.
    neutral = (
        alt.Chart(pairs[pairs["label"] == "neutral"])
        .mark_circle(size=20, color="gray", opacity=0.35)
        .encode(x=x, y=y, tooltip=tooltip)
    )
    points = (
        alt.Chart(tagged)
        .mark_circle(size=40, opacity=0.85)
        .encode(x=x, y=y, color=label_color, tooltip=tooltip)
    )
    if named_roles is None:
        # Most extreme points first, so the ends of the distribution keep their names when space runs out.
        candidates = tagged[
            tagged["role"].isin(sample_named_roles(desirability, name_fraction, seed))
        ]
        candidates = candidates.assign(
            extremity=candidates[stage_a].abs() + candidates[stage_b].abs()
        ).sort_values("extremity", ascending=False)
    else:
        candidates = pairs.set_index("role").loc[named_roles].reset_index()
    placed = place_names(
        candidates[stage_a].tolist(),
        candidates[stage_b].tolist(),
        candidates["role"].tolist(),
        [low, high],
        [low, high],
        460,
        460,
        obstacles=list(zip(tagged[stage_a], tagged[stage_b]))
        + list(zip(candidates[stage_a], candidates[stage_b])),
    )
    named = candidates.iloc[[i for i, _, _ in placed]].assign(
        label_x=[x for _, x, _ in placed],
        label_y=[y for _, _, y in placed],
        name_color=lambda df: df["label"].map({**LABEL_COLORS, "neutral": "dimgray"}),
    )
    # Named neutral roles stand out from the gray backdrop; names take literal colors, so neutral stays out of the label legend.
    named_neutral = (
        alt.Chart(named[named["label"] == "neutral"])
        .mark_circle(size=40, color="dimgray", opacity=0.85)
        .encode(x=x, y=y, tooltip=tooltip)
    )
    names = (
        alt.Chart(named)
        .mark_text(align="left", baseline="middle", fontSize=10, fontWeight="bold")
        .encode(
            x=alt.X("label_x:Q", scale=scale),
            y=alt.Y("label_y:Q", scale=scale),
            text="role",
            color=alt.Color("name_color:N", scale=None),
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"v": [low, high]}))
        .mark_line(color="lightgray", strokeDash=[4, 3])
        .encode(x="v:Q", y="v:Q")
    )
    return (
        (diagonal + neutral + named_neutral + points + names)
        .resolve_scale(color="independent")
        .properties(width=460, height=460, title=title)
    )


def slope_chart(
    positions: pd.DataFrame,
    stages: list[str],
    stage_a: str,
    stage_b: str,
    groups: pd.Series,
    group_scale: alt.Scale,
    legend_title: str,
    descriptions: pd.Series,
    axis_title: str,
    title: str,
    headings: bool,
) -> alt.HConcatChart:
    """Each persona's raw projection across stages, with the roles in `groups` highlighted.

    `positions` has columns stage, role and projection; `groups` maps highlighted roles to a group name.
    Line style shows the direction from stage_a to stage_b; `headings` names each group above its block of end labels.
    """
    last_index = len(stages) - 1
    lines = positions.assign(
        group=positions["role"].map(groups),
        description=positions["role"].map(descriptions),
        stage_index=positions["stage"].map({stage: i for i, stage in enumerate(stages)}),
    )
    highlighted_rows = lines[lines["group"].notna()]
    # A numeric x axis leaves room right of the last stage for connectors and labels.
    stage_names = "[" + ", ".join(f"'{stage}'" for stage in stages) + "]"
    x = alt.X(
        "stage_index:Q",
        title=None,
        scale=alt.Scale(domain=[-0.15, last_index + 0.75]),
        axis=alt.Axis(
            values=list(range(len(stages))), labelExpr=f"{stage_names}[datum.value]", grid=False
        ),
    )
    # Direction in color too: each group's own color when rising, a lighter shade of it when dropping.
    shade_domain, shade_range = [], []
    for group, color in zip(group_scale.domain, group_scale.range):
        rgb = np.array(to_rgb(color))
        shade_domain += [f"{group}, rising", f"{group}, dropping"]
        shade_range += [to_hex(tuple(rgb)), to_hex(tuple(rgb + (1 - rgb) * 0.35))]
    shade_color = alt.Color(
        "shade:N", scale=alt.Scale(domain=shade_domain, range=shade_range), title=legend_title
    )

    def panel(field: str, y_title: str) -> alt.LayerChart:
        # Direction over the selected pair.
        change = lines.pivot(index="role", columns="stage", values=field).eval(
            f"{stage_b} - {stage_a}"
        )
        direction = np.sign(change).map({1.0: "rising", -1.0: "dropping", 0.0: "rising"})
        highlighted = highlighted_rows.assign(direction=highlighted_rows["role"].map(direction))
        highlighted["shade"] = highlighted["group"] + ", " + highlighted["direction"]
        # Spread end labels by a fixed share of the panel's spread so close lines stay readable.
        gap = 0.13 * lines[field].std()
        last = highlighted[highlighted["stage"] == stages[-1]].sort_values(field).copy()
        last["text"] = last["direction"].map({"rising": "▲ ", "dropping": "▼ "}) + last["role"]
        last["label_y"] = spread_labels(last[field].tolist(), gap)
        last["label_x"] = last_index + 0.15
        wide = lines.pivot(index="role", columns="stage", values=field)
        last = last.assign(
            value_a=last["role"].map(wide[stage_a]),
            value_b=last["role"].map(wide[stage_b]),
            change=last["role"].map(change),
        )
        heading_rows = (
            last.groupby("group", as_index=False)["label_y"]
            .max()
            .assign(label_y=lambda d: d["label_y"] + 1.4 * gap, label_x=last_index + 0.15)
        )
        if not headings:
            heading_rows = heading_rows.iloc[0:0]

        y = alt.Y(f"{field}:Q", title=y_title)
        background = (
            alt.Chart(lines)
            .mark_line(color="lightgray", opacity=0.35, strokeWidth=1)
            .encode(x=x, y=y, detail="role", tooltip=["role", "description"])
        )
        foreground = (
            alt.Chart(highlighted)
            .mark_line(strokeWidth=2.5, point=True)
            .encode(
                x=x,
                y=y,
                detail="role",
                color=shade_color,
                strokeDash=alt.StrokeDash("direction:N", scale=DIRECTION_DASH, legend=None),
                tooltip=[
                    "role",
                    "description",
                    "direction",
                    alt.Tooltip(f"{field}:Q", format=".2f"),
                ],
            )
        )
        connectors = (
            alt.Chart(last)
            .mark_rule(strokeWidth=0.8, opacity=0.7)
            .encode(x=x, y=y, x2="label_x:Q", y2="label_y:Q", color=shade_color)
        )
        names = (
            alt.Chart(last)
            .mark_text(align="left", dx=3)
            .encode(
                x="label_x:Q",
                y="label_y:Q",
                text="text:N",
                color=shade_color,
                tooltip=[
                    "role",
                    alt.Tooltip("group:N", title="group"),
                    "description",
                    "direction",
                    alt.Tooltip("value_a:Q", title=stage_a, format=".2f"),
                    alt.Tooltip("value_b:Q", title=stage_b, format=".2f"),
                    alt.Tooltip("change:Q", title=f"{stage_a} → {stage_b}", format="+.2f"),
                ],
            )
        )
        # Fixed colors per heading: a color encoding here would merge with the shade scale.
        heading_marks = alt.layer(
            *[
                alt.Chart(heading_rows[heading_rows["group"] == group])
                .mark_text(align="left", dx=3, fontWeight="bold", fontSize=13, color=color)
                .encode(x="label_x:Q", y="label_y:Q", text="group:N")
                for group, color in zip(group_scale.domain, group_scale.range)
            ]
        )
        return (background + foreground + connectors + names + heading_marks).properties(
            width=400, height=480
        )

    # A hand-drawn key for line style: Vega-Lite's own strokeDash legend rendered without visible samples.
    key = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [1, 1, 0, 0],
            "direction": ["rising", "rising", "dropping", "dropping"],
        }
    )
    key_x = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, 3.2]))
    key_y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-0.6, 1.6]))
    key_lines = (
        alt.Chart(key)
        .mark_line(color="dimgray", strokeWidth=2.5)
        .encode(
            x=key_x,
            y=key_y,
            detail="direction:N",
            strokeDash=alt.StrokeDash("direction:N", scale=DIRECTION_DASH, legend=None),
        )
    )
    key_text = (
        alt.Chart(key[key["x"] == 1].assign(text=["▲ rising", "▼ dropping"]))
        .mark_text(align="left", dx=6)
        .encode(x=key_x, y=key_y, text="text:N")
    )
    direction_key = (key_lines + key_text).properties(
        width=120, height=60, title=alt.Title(f"{stage_a} → {stage_b}", fontSize=11, anchor="start")
    )

    return (
        alt.hconcat(
            panel("projection", f"raw {axis_title}"),
            direction_key,
        )
        .configure_view(stroke=None)
        .properties(title=title)
    )


def top_mover_groups(positions: pd.DataFrame, stage_a: str, stage_b: str, n: int) -> pd.Series:
    """The n roles whose z-score rose most and the n that dropped most from stage_a to stage_b."""
    change = positions.pivot(index="role", columns="stage", values="z").eval(
        f"{stage_b} - {stage_a}"
    )
    return pd.concat(
        [
            pd.Series("rose", index=change.nlargest(n).index),
            pd.Series("dropped", index=change.nsmallest(n).index),
        ]
    )


def top_mover_labels(
    positions: pd.DataFrame, stage_a: str, stage_b: str, desirability: pd.DataFrame, n: int
) -> pd.Series:
    """For each of desirable and undesirable, the n roles whose z-score changed most from stage_a to stage_b, mapped to their label."""
    change = (
        positions.pivot(index="role", columns="stage", values="z")
        .eval(f"{stage_b} - {stage_a}")
        .dropna()
        .abs()
    )
    labels = desirability.set_index("role")["label"].reindex(change.index)
    return pd.concat(
        [
            pd.Series(label, index=change[labels == label].nlargest(n).index)
            for label in LABEL_COLORS
        ]
    )


def style_by_label(table: pd.DataFrame, desirability: pd.DataFrame):
    """Color each role name by its desirability label, as in the plots; neutral roles stay plain."""
    colors = desirability.set_index("role")["label"].map(LABEL_COLORS).dropna()
    return (
        table.style.hide(axis="index")
        .format(precision=2)
        .map(
            lambda role: f"color: {colors[role]}; font-weight: bold" if role in colors else "",
            subset=["role"],
        )
    )
