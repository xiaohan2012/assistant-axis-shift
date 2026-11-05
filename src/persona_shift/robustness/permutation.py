"""Permutation test of a two-group mean difference: shuffle the labels to get its null distribution."""

import numpy as np


def mean_difference(values: np.ndarray, labels: np.ndarray) -> float:
    """Mean of `values` where the boolean `labels` is True minus their mean where it is False.

    In our case,

    - values can be the per-role projection on the default assistant axis
    - labels can be whether is the role is desirable one or not
    """
    return float(values[labels].mean() - values[~labels].mean())


def permutation_test(
    values: np.ndarray, labels: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> tuple[float, np.ndarray, float]:
    """Test whether the mean difference between two labelled groups (e.g., good roles vs bad roles) is larger than chance.

    Shuffles the boolean `labels` n_permutations times and recomputes the difference each time.

    Returns:
    - observed: mean difference under the real labels.
    - null: mean differences under shuffled labels, shape (n_permutations,).
    - p_value: two-sided; share of shuffles at least as extreme as observed, with the +1 correction.
    """
    observed = mean_difference(values, labels)
    null = np.array(
        [mean_difference(values, rng.permutation(labels)) for _ in range(n_permutations)]
    )
    p_value = (1 + (np.abs(null) >= abs(observed)).sum()) / (1 + n_permutations)
    return observed, null, float(p_value)
