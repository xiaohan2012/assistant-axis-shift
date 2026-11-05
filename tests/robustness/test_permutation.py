import numpy as np
import pytest

from persona_shift.robustness.permutation import mean_difference, permutation_test


class TestMeanDifference:
    """Test function mean_difference"""

    def test_true_mean_minus_false_mean(self) -> None:
        assert mean_difference(
            np.array([3.0, 5.0, 1.0]), np.array([True, True, False])
        ) == pytest.approx(3.0)


class TestPermutationTest:
    """Test function permutation_test"""

    def test_separated_groups_are_significant(self) -> None:
        values = np.r_[np.full(20, 10.0), np.zeros(20)]
        labels = np.r_[np.ones(20, dtype=bool), np.zeros(20, dtype=bool)]

        observed, null, p_value = permutation_test(values, labels, 999, np.random.default_rng(0))

        assert observed == pytest.approx(10.0)
        assert null.shape == (999,)
        assert p_value == pytest.approx(1 / 1000)

    def test_unrelated_labels_are_not_significant(self) -> None:
        rng = np.random.default_rng(0)
        _, _, p_value = permutation_test(rng.normal(size=200), rng.random(200) < 0.5, 999, rng)
        assert p_value > 0.05
