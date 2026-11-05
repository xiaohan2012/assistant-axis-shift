import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.linalg import subspace_angles
from scipy.stats import ortho_group

from persona_shift.persona_axes import (
    axis_projections,
    flip_pairs,
    judge_score_3_response_ids,
    load_pass_rates,
    pair_table,
    participation_ratio,
    principal_vector_pairs,
    project,
    shared_roles,
    sum_activations_by_question,
    variance_share,
    zscore,
)
from persona_shift.persona_charts import (
    density_curve,
    place_names,
    sample_named_roles,
    spread_labels,
    top_mover_labels,
)

N_ROLES = 30
DIM = 50


def random_matrix(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(N_ROLES, DIM))


class TestLoadPassRates:
    """Test function load_pass_rates"""

    def test_share_of_score_3_among_judged_responses(self, tmp_path: Path) -> None:
        scores_dir = tmp_path / "sft-sysprompt" / "scores-glm-5.3-flash"
        scores_dir.mkdir(parents=True)
        scores = {"pos_p0_q0": 3, "pos_p0_q1": 2, "pos_p0_q2": 3, "pos_p0_q3": 0}
        (scores_dir / "poet.json").write_text(json.dumps(scores))
        rates = load_pass_rates(tmp_path).set_index("role")
        assert rates.loc["poet", "stage"] == "SFT"
        assert rates.loc["poet", "n_judged"] == 4
        assert rates.loc["poet", "pass_rate"] == pytest.approx(0.5)


class TestJudgeScore3ResponseIds:
    """Test function judge_score_3_response_ids"""

    def test_keeps_score_3_only(self) -> None:
        assert judge_score_3_response_ids({"poet": {"pos_p0_q0": 3, "pos_p0_q1": 2}}) == {
            "poet": {"pos_p0_q0"}
        }


class TestSumActivationsByQuestion:
    """Test function sum_activations_by_question"""

    def write_role(self, folder: Path, role: str, keys: list[str], acts: np.ndarray) -> None:
        torch.save(
            {"keys": keys, "acts": torch.tensor(acts, dtype=torch.bfloat16)}, folder / f"{role}.pt"
        )

    def test_sums_kept_responses_per_question(self, tmp_path: Path) -> None:
        folder = tmp_path / "sft-sysprompt" / "activations-layer16"
        folder.mkdir(parents=True)
        keys = ["pos_p0_q0", "pos_p1_q0", "pos_p0_q5"]
        acts = np.array([[1.0] * DIM, [2.0] * DIM, [4.0] * DIM])
        self.write_role(folder, "poet", keys, acts)
        self.write_role(folder, "default", keys, acts)

        roles, sums, counts = sum_activations_by_question(
            tmp_path, "SFT", 16, {"poet": {"pos_p0_q0", "pos_p0_q5"}}
        )

        assert roles == ["poet", "default"]
        np.testing.assert_allclose(sums[0, 0], [1.0] * DIM)
        np.testing.assert_allclose(sums[0, 5], [4.0] * DIM)
        np.testing.assert_allclose(sums[1, 0], [3.0] * DIM)
        assert counts[0].sum() == 2
        assert counts[1].sum() == 3


class TestAxisProjections:
    """Test function axis_projections"""

    def test_roles_relative_to_their_mean_along_default_direction(self) -> None:
        role_vectors = np.array([[1.0, 0.0], [-1.0, 2.0]])
        default_vector = np.array([3.0, 2.0])

        result = axis_projections(role_vectors, default_vector)

        assert result.mean() == pytest.approx(0.0)
        assert result[0] == pytest.approx(
            project(role_vectors[0], np.array([3.0, 1.0]), np.array([0.0, 1.0]))
        )


class TestSharedRoles:
    """Test function shared_roles"""

    def test_keeps_common_roles_in_one_order(self) -> None:
        roles, matrix_a, matrix_b = shared_roles(
            ["poet", "tutor", "translator"],
            np.array([[1.0], [2.0], [3.0]]),
            ["tutor", "poet"],
            np.array([[20.0], [10.0]]),
        )
        assert roles == ["poet", "tutor"]
        np.testing.assert_allclose(matrix_a, [[1.0], [2.0]])
        np.testing.assert_allclose(matrix_b, [[10.0], [20.0]])


class TestProject:
    """Test function project"""

    def test_scalar_projection_onto_unit_direction(self) -> None:
        points = np.array([[3.0, 4.0], [1.0, 1.0]])
        np.testing.assert_allclose(
            project(points, np.array([0.0, 2.0]), np.array([1.0, 1.0])), [3.0, 0.0]
        )


class TestParticipationRatio:
    """Test function participation_ratio"""

    @pytest.mark.parametrize("n_dims", [1, 3, 7])
    def test_equal_variance_dimensions_count_fully(self, n_dims: int) -> None:
        # Rows +-e_i for the first n_dims axes: n_dims orthogonal directions with equal variance.
        eye = np.eye(DIM)[:n_dims]
        assert participation_ratio(np.vstack([eye, -eye])) == pytest.approx(n_dims)

    def test_invariant_to_rotation_and_offset(self) -> None:
        x = random_matrix(0)
        rotated = x @ ortho_group.rvs(DIM, random_state=1) + 5.0
        assert participation_ratio(rotated) == pytest.approx(participation_ratio(x))


class TestPrincipalVectorPairs:
    """Test function principal_vector_pairs"""

    def test_cosines_match_scipy_subspace_angles(self) -> None:
        x, y = random_matrix(0), random_matrix(1)
        cosines, _, _ = principal_vector_pairs(x, y, k=N_ROLES - 1)
        xc, yc = x - x.mean(0), y - y.mean(0)
        expected = np.sort(np.cos(subspace_angles(xc.T, yc.T)))[::-1]
        np.testing.assert_allclose(cosines, expected, atol=1e-8)

    def test_pairs_are_orthonormal_with_cosines_between_them(self) -> None:
        cosines, u, v = principal_vector_pairs(random_matrix(0), random_matrix(1), k=5)
        np.testing.assert_allclose(u.T @ u, np.eye(5), atol=1e-10)
        np.testing.assert_allclose(v.T @ v, np.eye(5), atol=1e-10)
        np.testing.assert_allclose(np.diag(u.T @ v), cosines, atol=1e-10)

    def test_identical_matrices_have_cosines_one(self) -> None:
        x = random_matrix(0)
        cosines, _, _ = principal_vector_pairs(x, x, k=5)
        np.testing.assert_allclose(cosines, np.ones(5), atol=1e-10)


class TestVarianceShare:
    """Test function variance_share"""

    def test_shares_over_a_full_basis_sum_to_one(self) -> None:
        x = random_matrix(0)
        basis = np.linalg.svd(x - x.mean(0), full_matrices=False)[2].T
        assert variance_share(x, basis).sum() == pytest.approx(1.0)


class TestFlipPairs:
    """Test function flip_pairs"""

    def test_flips_only_pairs_with_negative_scores(self) -> None:
        u, v = np.eye(3), 2 * np.eye(3)
        flipped_u, flipped_v = flip_pairs(u, v, np.array([1.0, -0.5, 0.0]))
        np.testing.assert_allclose(flipped_u, np.diag([1.0, -1.0, 1.0]))
        np.testing.assert_allclose(flipped_v, np.diag([2.0, -2.0, 2.0]))


class TestZscore:
    """Test function zscore"""

    def test_mean_zero_std_one(self) -> None:
        z = zscore(np.array([1.0, 2.0, 6.0]))
        assert z.mean() == pytest.approx(0.0)
        assert z.std() == pytest.approx(1.0)


class TestPairTable:
    """Test function pair_table"""

    @property
    def positions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stage": ["A", "A", "A", "B", "B", "B"],
                "role": ["poet", "tutor", "default", "poet", "tutor", "default"],
                "is_default": [False, False, True, False, False, True],
                "projection": [0.5, -0.5, 2.0, -1.0, 1.0, 3.0],
            }
        )

    def test_ranks_changes_labels_and_descriptions(self) -> None:
        desirability = pd.DataFrame({"role": ["tutor"], "label": ["desirable"]})
        descriptions = pd.Series({"poet": "writes verse", "tutor": "teaches"})
        table = pair_table(self.positions, "A", "B", desirability, descriptions).set_index("role")
        assert table.loc["poet", "A_rank"] == 2
        assert table.loc["poet", "B_rank"] == 1
        assert table.loc["tutor", "rank_change"] == 1
        assert table.loc["tutor", "projection_change"] == pytest.approx(1.5)
        assert table.loc["poet", "label"] == "neutral"
        assert table.loc["tutor", "description"] == "teaches"
        assert "default" not in table.index

    def test_other_value_column(self) -> None:
        positions = self.positions.rename(columns={"projection": "pass_rate"})
        table = pair_table(
            positions,
            "A",
            "B",
            pd.DataFrame({"role": [], "label": []}),
            pd.Series(dtype=str),
            value="pass_rate",
        ).set_index("role")
        assert table.loc["tutor", "pass_rate_change"] == pytest.approx(1.5)

    def test_roles_missing_from_a_stage_are_dropped(self) -> None:
        # A role without a vector in one stage, as translator in the 275-role SFT run.
        positions = pd.concat(
            [
                self.positions,
                pd.DataFrame(
                    {
                        "stage": ["B"],
                        "role": ["translator"],
                        "is_default": [False],
                        "projection": [0.0],
                    }
                ),
            ]
        )
        table = pair_table(
            positions, "A", "B", pd.DataFrame({"role": [], "label": []}), pd.Series(dtype=str)
        )
        assert sorted(table["role"]) == ["poet", "tutor"]


class TestSampleNamedRoles:
    """Test function sample_named_roles"""

    @property
    def desirability(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "role": [f"good{i}" for i in range(10)] + [f"bad{i}" for i in range(5)] + ["tree"],
                "label": ["desirable"] * 10 + ["undesirable"] * 5 + ["neutral"],
            }
        )

    def test_samples_the_fraction_of_each_group_and_no_neutral(self) -> None:
        named = sample_named_roles(self.desirability, name_fraction=0.4, seed=0)
        assert sum(role.startswith("good") for role in named) == 4
        assert sum(role.startswith("bad") for role in named) == 2
        assert "tree" not in named

    def test_same_seed_same_roles(self) -> None:
        assert sample_named_roles(self.desirability, seed=3) == sample_named_roles(
            self.desirability, seed=3
        )


class TestPlaceNames:
    """Test function place_names"""

    # 100 x 100 data units on a 100 x 100 px chart, so one unit is one pixel.
    @staticmethod
    def place(xs: list[float], ys: list[float], names: list[str]) -> list[tuple[int, float, float]]:
        return place_names(xs, ys, names, [0, 100], [0, 100], 100, 100, char_px=5, line_px=10)

    def test_free_name_goes_right_of_its_point(self) -> None:
        assert self.place([20], [50], ["ab"]) == [(0, 26, 50)]

    def test_second_name_avoids_the_first(self) -> None:
        # Both points share a spot; the second name takes the next free candidate, left of its point.
        placed = self.place([40, 40], [50, 50], ["abcd", "abcd"])
        assert placed == [(0, 46, 50), (1, 14, 50)]

    def test_name_avoids_obstacle_points(self) -> None:
        # A point just right of the first spot pushes the name to the left of its own point.
        placed = place_names(
            [40],
            [50],
            ["abcd"],
            [0, 100],
            [0, 100],
            100,
            100,
            char_px=5,
            line_px=10,
            obstacles=[(50, 50)],
        )
        assert placed == [(0, 14, 50)]

    def test_name_without_room_is_dropped(self) -> None:
        # A name wider than the chart fits nowhere.
        assert self.place([50], [50], ["x" * 30]) == []


class TestDensityCurve:
    """Test function density_curve"""

    def test_area_equals_count_times_bin_width(self) -> None:
        # Integrated over a wide grid, counts per bin of width 0.2 sum back to n roles: area = n * 0.2.
        values = np.random.default_rng(0).normal(size=40)
        grid = np.linspace(-10, 10, 4001)
        area = np.trapezoid(density_curve(values, grid, bin_step=0.2), grid)
        assert area == pytest.approx(40 * 0.2, rel=1e-3)


class TestTopMoverLabels:
    """Test function top_mover_labels"""

    def test_largest_absolute_change_per_label(self) -> None:
        positions = pd.DataFrame(
            {
                "stage": ["A"] * 5 + ["B"] * 5,
                "role": ["tutor", "coach", "spy", "virus", "tree"] * 2,
                "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.9, 0.5, -0.1, 3.0],
            }
        )
        desirability = pd.DataFrame(
            {
                "role": ["tutor", "coach", "spy", "virus", "tree"],
                "label": ["desirable", "desirable", "undesirable", "undesirable", "neutral"],
            }
        )
        movers = top_mover_labels(positions, "A", "B", desirability, n=1)
        assert movers.to_dict() == {"coach": "desirable", "spy": "undesirable"}


class TestSpreadLabels:
    """Test function spread_labels"""

    def test_separated_labels_stay_put(self) -> None:
        assert spread_labels([0.0, 1.0, 2.0], gap=0.5) == pytest.approx([0.0, 1.0, 2.0])

    def test_crowded_block_is_centered_on_its_mean(self) -> None:
        assert spread_labels([1.0, 1.0, 1.0], gap=0.5) == pytest.approx([0.5, 1.0, 1.5])

    def test_labels_end_at_least_gap_apart(self) -> None:
        positions = spread_labels([0.0, 0.1, 0.15, 0.9, 0.95], gap=0.2)
        assert np.diff(positions).min() >= 0.2 - 1e-12
