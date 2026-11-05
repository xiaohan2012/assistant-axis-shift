import numpy as np
import pytest

from persona_shift.persona_axes import N_QUESTIONS, axis_projections
from persona_shift.robustness.split_half import full_data_se, reliability, split_half

DIM = 8
N_ROLES = 6


class TestSplitHalf:
    """Test function split_half"""

    def test_halves_match_direct_recompute(self) -> None:
        sums = np.random.default_rng(0).normal(size=(N_ROLES, N_QUESTIONS, DIM))
        counts = np.random.default_rng(1).integers(1, 6, size=(N_ROLES, N_QUESTIONS)).astype(float)

        a, b = split_half(sums, counts, 3, np.random.default_rng(2))

        rng = np.random.default_rng(2)
        for repeat in range(3):
            half = rng.permutation(N_QUESTIONS) < N_QUESTIONS // 2
            for result, questions in [(a, half), (b, ~half)]:
                vectors = sums[:, questions].sum(axis=1) / counts[:, questions].sum(axis=1)[:, None]
                np.testing.assert_allclose(
                    result[repeat], axis_projections(vectors[:-1], vectors[-1]), rtol=1e-6
                )

    def test_identical_questions_give_identical_halves(self) -> None:
        per_role = np.random.default_rng(0).normal(size=(N_ROLES, 1, DIM))
        sums = np.repeat(per_role, N_QUESTIONS, axis=1)

        a, b = split_half(sums, np.ones((N_ROLES, N_QUESTIONS)), 5, np.random.default_rng(0))

        assert a.shape == (5, N_ROLES - 1)
        np.testing.assert_allclose(a, b, atol=1e-6)


class TestReliability:
    """Test function reliability"""

    def test_correlation_per_repeat(self) -> None:
        a = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        b = np.array([[2.0, 4.0, 6.0], [3.0, 2.0, 1.0]])
        np.testing.assert_allclose(reliability(a, b), [1.0, -1.0])


class TestFullDataSe:
    """Test function full_data_se"""

    def test_halves_the_spread_of_differences(self) -> None:
        differences = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        assert full_data_se(differences)[0] == pytest.approx(differences.std(ddof=1) / 2)
