import numpy as np

from src.agreement import top_set, topx_jaccard
from src.stats import holm, paired_comparison, rank_biserial


def test_paired_comparison_all_negative():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], float)
    c = paired_comparison(a - 0.1 * np.arange(1, 9), a)
    assert c["wins_A"] == 8 and c["rank_biserial"] == -1.0
    assert abs(c["wilcoxon_p"] - 2 / 2**8) < 1e-12


def test_paired_comparison_with_zero_differences():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.5, 4.5])
    c = paired_comparison(a, b)
    assert c["ties"] == 2 and c["n_nonzero"] == 2
    assert abs(c["wilcoxon_p"] - 0.5) < 1e-12  # exact test on the two non-zero pairs


def test_rank_biserial_and_holm():
    assert rank_biserial([1, 2, -3]) == (3 - 3) / 6
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_top_sets_are_tie_inclusive():
    s = np.array([5, 5, 5, 1, 1, 1, 1, 1, 1, 1])
    assert top_set(s, 0.1) == {0, 1, 2}
    assert topx_jaccard(s, s.copy(), 0.1) == 1.0
    be = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert top_set(be, 0.2) == {0}  # lowest tier never included
