"""Task 3: select_guess policy tests.

Hand-built 5-word toy list in this order (indices 0-4), matrix built via the
scalar feedback_pattern:

    0 AAAAA  1 BBBBB  2 CCCCC  3 DEEED  4 EABCD

The toy matrix (see test_select_guess.py fixture): EABCD's row is
[6, 18, 54, 163, 242] — the only guess whose buckets are all distinct.
"""

import numpy as np
import pytest

import wordle_core as wc

TOY = ["AAAAA", "BBBBB", "CCCCC", "DEEED", "EABCD"]


@pytest.fixture(scope="module")
def toy_matrix():
    return np.array(
        [[wc.feedback_pattern(guess, solution) for solution in TOY] for guess in TOY],
        dtype=np.uint8,
    )


def test_all_legal_prefers_split(toy_matrix):
    # S = all 5. Only EABCD splits every pair (five singleton buckets):
    # entropy log2(5) vs 1.37... for the other four, unique argmax.
    S = np.arange(5)
    assert wc.select_guess(S, None, toy_matrix) == 4


def test_tie_prefers_member(toy_matrix):
    # S = {1 (BBBBB), 2 (CCCCC)}. BBBBB, CCCCC and EABCD all split 1/1
    # (entropy 1.0, expected remaining 1.0); AAAAA and DEEED give all-gray
    # (entropy 0). Member tie-break + list order -> BBBBB, and DEEED must
    # NOT tie — the member branch beats the order branch with non-members
    # present.
    S = np.array([1, 2])
    assert wc.select_guess(S, None, toy_matrix) == 1


def test_hard_mode_respects_legal(toy_matrix):
    # Same S = {1, 2}, but BBBBB is illegal. Among legal candidates CCCCC
    # (member) and EABCD (non-member) tie at 1.0/1.0; the member branch
    # picks CCCCC.
    S = np.array([1, 2])
    legal = np.ones(5, dtype=bool)
    legal[1] = False
    assert wc.select_guess(S, legal, toy_matrix) == 2


def test_single_solution_legal_returns_member(toy_matrix):
    # S = {2} (CCCCC), legal: the member is returned immediately, whatever
    # entropy the other candidates score (here it is 0 everywhere anyway).
    S = np.array([2])
    assert wc.select_guess(S, None, toy_matrix) == 2


def test_single_solution_illegal_falls_back(toy_matrix):
    # S = {2} but CCCCC illegal and EABCD the only legal candidate. Every
    # metric ties (entropy 0, expected remaining 1 for a single-solution
    # set), the member tie-break has no member among the legal, so the
    # chain falls through to list order -> the one legal guess.
    S = np.array([2])
    legal = np.zeros(5, dtype=bool)
    legal[4] = True
    assert wc.select_guess(S, legal, toy_matrix) == 4


def test_no_legal_candidates(toy_matrix):
    S = np.array([2])
    with pytest.raises(ValueError, match="no legal candidates"):
        wc.select_guess(S, np.zeros(5, dtype=bool), toy_matrix)
