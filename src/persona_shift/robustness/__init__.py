"""Checks that separate real persona shifts along the assistant axis from noise.

- `permutation`: is the good-vs-bad contrast larger than a random grouping of the same roles gives?
- `split_half`: is a shift larger than the sampling noise of the role vectors?

Both run on full or matched-sample vectors from `persona_axes.sum_activations_by_question`; the two differ only in which responses are kept.
"""
