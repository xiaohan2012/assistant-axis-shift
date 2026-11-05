"""Split-half: rebuild the role projections from two random halves of one stage's questions.

The halves differ only by sampling noise, so one run gives two things:
- standard error: from the spread of half A minus half B, see `full_data_se`.
- reliability: correlation of role positions between the halves, see `reliability`.

A bootstrap over questions would also give the standard error, without the /2 correction, but not the reliability.
Split-half is used because it gives both.
"""

import numpy as np

from persona_shift import persona_axes as pa


def split_half(
    sums: np.ndarray, counts: np.ndarray, n_repeats: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild every role's projection from two random halves of the questions, n_repeats times.

    Within one stage, the halves differ only by sampling noise, so their spread measures the noise of a projection.

    Note that questions, not responses, are split, because the five system-prompt answers to one question are correlated.

    Args:
    - sums, counts: per-question activation sums and response counts from `sum_activations_by_question`; one row per role, then a last row for `default` (the Assistant, no role prompt).
    - n_repeats: number of random splits.
    - rng: draws the splits; one seed gives the same splits for both stages and for full and matched, so their runs pair up repeat by repeat.

    Returns:
    - a, b: projections from each half, shape (n_repeats, n_roles), without `default`.
    """
    by_question = sums.transpose(0, 2, 1)
    total_sums, total_counts = sums.sum(axis=1), counts.sum(axis=1)
    a, b = [], []
    for _ in range(n_repeats):
        half = (rng.permutation(pa.N_QUESTIONS) < pa.N_QUESTIONS // 2).astype(sums.dtype)
        # One pass over the sums per repeat: the other half is the total minus this one.
        half_sums, half_counts = by_question @ half, counts @ half
        for out, vectors in [
            (a, half_sums / half_counts[:, None]),
            (b, (total_sums - half_sums) / (total_counts - half_counts)[:, None]),
        ]:
            out.append(pa.axis_projections(vectors[:-1], vectors[-1]))
    return np.array(a), np.array(b)


def reliability(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation over roles between the two halves, per repeat."""
    return np.array([np.corrcoef(row_a, row_b)[0, 1] for row_a, row_b in zip(a, b)])


def full_data_se(differences: np.ndarray) -> np.ndarray:
    """Standard error of a full-data value from its half-minus-half differences.

    A half has half the data, so twice the variance, and a difference of two halves doubles it again; hence sd / 2.
    """
    return differences.std(axis=0, ddof=1) / 2
