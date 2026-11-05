"""Load persona vectors and place personas along directions in activation space.

A direction is either a stage's assistant axis or a principal vector shared between two stages.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

STAGES = {"SFT": "sft-sysprompt", "DPO": "dpo-sysprompt", "RLVR": "rlvr-sysprompt"}
# Row order of the stored (3, 4096) tensors, set by LAYERS in scripts/pod/run_pilot.sh.
LAYERS = [8, 16, 24]
JUDGE_TAG = "glm-5.3-flash"
# Questions in external/assistant-axis/data/extraction_questions.jsonl; response keys are pos_p{prompt}_q{question}.
N_QUESTIONS = 240


def load_vectors(data_dir: Path) -> pd.DataFrame:
    """One row per (stage, role, layer), with the role's vector; `default` is the Assistant itself."""
    rows = []
    for stage, run in STAGES.items():
        for path in sorted((data_dir / run / f"vectors-{JUDGE_TAG}").glob("*.pt")):
            vector = torch.load(path, weights_only=False)["vector"].float().numpy()
            for i, layer in enumerate(LAYERS):
                rows.append(
                    {"stage": stage, "role": path.stem, "layer": layer, "vector": vector[i]}
                )
    vectors = pd.DataFrame(rows)
    vectors["is_default"] = vectors["role"] == "default"
    return vectors


def load_axes(data_dir: Path) -> pd.DataFrame:
    """One row per (stage, layer) with the assistant axis written by the pipeline's step 5."""
    rows = []
    for stage, run in STAGES.items():
        axis = torch.load(data_dir / run / f"axis-{JUDGE_TAG}.pt").float().numpy()
        for i, layer in enumerate(LAYERS):
            rows.append({"stage": stage, "layer": layer, "axis": axis[i]})
    return pd.DataFrame(rows)


def load_pass_rates(data_dir: Path) -> pd.DataFrame:
    """One row per (stage, role) with the share of judged responses scored 3, fully in role.

    Score 3 is the rule 4_vectors.py uses to build each role's vector.
    """
    rows = []
    for stage, run in STAGES.items():
        for path in sorted((data_dir / run / f"scores-{JUDGE_TAG}").glob("*.json")):
            scores = pd.read_json(path, typ="series")
            rows.append(
                {
                    "stage": stage,
                    "role": path.stem,
                    "n_judged": len(scores),
                    "n_pass": int((scores == 3).sum()),
                }
            )
    rates = pd.DataFrame(rows)
    rates["pass_rate"] = rates["n_pass"] / rates["n_judged"]
    return rates


def load_judge_scores(data_dir: Path, stage: str) -> dict[str, dict[str, int]]:
    """Judge scores (0-3) per role, keyed by response id such as `pos_p0_q0`."""
    folder = data_dir / STAGES[stage] / f"scores-{JUDGE_TAG}"
    return {
        path.stem: pd.read_json(path, typ="series").to_dict()
        for path in sorted(folder.glob("*.json"))
    }


def judge_score_3_response_ids(scores: dict[str, dict[str, int]]) -> dict[str, set[str]]:
    """Per role, the ids of responses the judge scored 3 (fully in role), the ones 4_vectors.py averages."""
    return {
        role: {response_id for response_id, score in role_scores.items() if score == 3}
        for role, role_scores in scores.items()
    }


def sum_activations_by_question(
    data_dir: Path, stage: str, layer: int, keep: dict[str, set[str]]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sum each role's kept response activations per question, over the five system prompts.

    A role vector for any subset of questions is then a sum over those questions divided by the count, without reloading activations.

    Args:
    - keep: per role, the ids of the responses to keep; `default` keeps all of its responses, as in 4_vectors.py.

    Returns:
    - roles: the roles in `keep`, sorted, then `default` last.
    - sums: activation sums, shape (n_roles, N_QUESTIONS, hidden), rows in `roles` order.
    - counts: number of kept responses, shape (n_roles, N_QUESTIONS).
    """
    folder = data_dir / STAGES[stage] / f"activations-layer{layer}"
    roles = sorted(keep) + ["default"]
    sums, counts = [], []
    for role in roles:
        stored = torch.load(folder / f"{role}.pt")
        kept = np.array([role == "default" or key in keep[role] for key in stored["keys"]])
        questions = np.array([int(key.rsplit("_q", 1)[1]) for key in stored["keys"]])
        one_hot = (questions[kept][None, :] == np.arange(N_QUESTIONS)[:, None]).astype(np.float32)
        sums.append(one_hot @ stored["acts"].float().numpy()[kept])
        counts.append(one_hot.sum(axis=1))
    return roles, np.stack(sums), np.stack(counts)


def load_desirability(path: Path) -> pd.DataFrame:
    """Columns role, label (desirable, undesirable, neutral) and borderline."""
    labels = pd.read_json(path, orient="index")
    return labels.rename_axis("role").reset_index()[["role", "label", "borderline"]]


def load_role_descriptions(path: Path) -> pd.Series:
    """One-line description per role, indexed by role."""
    return pd.read_json(path, typ="series")


def stage_matrix(
    vectors: pd.DataFrame, stage: str, layer: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The roles, their (n_roles, hidden) matrix, and the default vector, for one stage and layer."""
    selected = vectors[(vectors["stage"] == stage) & (vectors["layer"] == layer)]
    roles = selected[~selected["is_default"]]
    default = selected.loc[selected["is_default"], "vector"]
    return roles["role"].tolist(), np.stack(roles["vector"]), default.iloc[0]


def shared_roles(
    roles_a: list[str], matrix_a: np.ndarray, roles_b: list[str], matrix_b: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Keep the roles present in both stages, in one order, so row i of both matrices is the same persona.

    A role can lack a vector in one stage when too few of its responses were judged fully in role.
    """
    common = sorted(set(roles_a) & set(roles_b))
    rows_a = [roles_a.index(role) for role in common]
    rows_b = [roles_b.index(role) for role in common]
    return common, matrix_a[rows_a], matrix_b[rows_b]


def project(points: np.ndarray, direction: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Scalar projection of each point, relative to origin, onto the unit direction."""
    return (points - origin) @ (direction / np.linalg.norm(direction))


def axis_projections(role_vectors: np.ndarray, default_vector: np.ndarray) -> np.ndarray:
    """Each role's projection on the assistant axis these vectors define.

    Same convention as the assistant_axis notebook: axis = default minus the mean role vector, origin = the mean role vector.
    """
    origin = role_vectors.mean(axis=0)
    return project(role_vectors, default_vector - origin, origin)


def participation_ratio(matrix: np.ndarray) -> float:
    """Effective number of dimensions of the mean-centered rows: (sum of variances)^2 / sum of squared variances."""
    singular_values = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
    variances = singular_values**2
    return float(variances.sum() ** 2 / (variances**2).sum())


def principal_vector_pairs(
    x: np.ndarray, y: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal angles between the top-k principal subspaces of two mean-centered persona matrices.

    Returns the cosines, largest first, and the matched principal vectors as columns of u (in x's space) and v (in y's space).
    Each pair (u_i, v_i) is defined up to a joint sign flip.
    """
    basis_x = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)[2][:k].T
    basis_y = np.linalg.svd(y - y.mean(axis=0), full_matrices=False)[2][:k].T
    left, cosines, right_t = np.linalg.svd(basis_x.T @ basis_y)
    return np.clip(cosines, 0.0, 1.0), basis_x @ left, basis_y @ right_t.T


def variance_share(matrix: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Share of the mean-centered rows' total variance that lies along each unit-length column of directions."""
    centered = matrix - matrix.mean(axis=0)
    return ((centered @ directions) ** 2).sum(axis=0) / (centered**2).sum()


def flip_pairs(u: np.ndarray, v: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip each pair (u_i, v_i) whose orientation score is negative, so the chosen side reads as positive."""
    signs = np.where(scores < 0, -1.0, 1.0)
    return u * signs, v * signs


def zscore(values: np.ndarray) -> np.ndarray:
    """Values rescaled to mean 0 and standard deviation 1."""
    return (values - values.mean()) / values.std()


def pair_table(
    positions: pd.DataFrame,
    stage_a: str,
    stage_b: str,
    desirability: pd.DataFrame,
    descriptions: pd.Series,
    value: str = "projection",
) -> pd.DataFrame:
    """One row per role with its value and rank in both stages, the changes, its label and its description.

    `positions` has columns stage, role, is_default and `value`; the default row and roles missing from either stage are dropped.
    Rank 1 is the lowest value.
    """
    table = (
        positions[~positions["is_default"]]
        .pivot(index="role", columns="stage", values=value)
        .dropna(subset=[stage_a, stage_b])
        .reset_index()
    )
    for stage in [stage_a, stage_b]:
        table[f"{stage}_rank"] = table[stage].rank().astype(int)
    table[f"{value}_change"] = table[stage_b] - table[stage_a]
    table["rank_change"] = table[f"{stage_b}_rank"] - table[f"{stage_a}_rank"]
    table = table.merge(desirability[["role", "label"]], on="role", how="left").fillna(
        {"label": "neutral"}
    )
    table["description"] = table["role"].map(descriptions)
    return table
