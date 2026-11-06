"""Task 4: expected_and_worst_turns recursion tests.

Toy matrix, scalar-built. The candidate space is Task 3's five-word list

    0 AAAAA  1 BBBBB  2 CCCCC  3 DEEED  4 EABCD

while the pinned solution sets are the first-three-word subsets ({0,1} and
{0,1,2}) of the brief — and the pinned policy choices need the two extra
non-member candidates: EABCD ties at entropy 1.0 on S = {0,1} (lost to the
member tie-break) and is the only perfect 1/1/1 splitter for S = {0,1,2}.
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


def test_turns_two_words(toy_matrix):
    # Verified with select_guess: over candidates AAAAA, BBBBB, EABCD (all
    # 1/1 splitters, entropy 1.0, expected remaining 1.0) the member
    # tie-break plus list order pick index 0. AAAAA buckets S = {0, 1} into
    # all-green {0} (solved this turn) and all-gray {1} (solved in one more),
    # so E = 1 + 1/2 * 1.0 = 1.5 and W = 1 + max(1, 0) = 2.
    memo = {}
    assert wc.select_guess(np.array([0, 1]), None, toy_matrix) == 0
    assert wc.expected_and_worst_turns(np.array([0, 1]), None, toy_matrix, memo) == (1.5, 2)


def test_turns_three_words_perfect_split(toy_matrix):
    # Verified with select_guess: EABCD (index 4, a non-member) is the only
    # guess whose buckets are singletons on S = {0,1,2} (patterns 6/18/54,
    # a 1/1/1 split); every member leaves a 2-bucket. Every bucket is then
    # solved in one more turn: E = 1 + 1/3*(1+1+1) = 2.0, W = 1 + 1 = 2.
    memo = {}
    S = np.array([0, 1, 2])
    g = wc.select_guess(S, None, toy_matrix)
    assert g == 4
    assert sorted(toy_matrix[g, S].tolist()) == [6, 18, 54]  # 1/1/1 buckets
    # E is exactly 2.0 in exact arithmetic (1 + 3 * 1/3); compare via approx
    # because the recursion accumulates it as floats.
    e, w = wc.expected_and_worst_turns(S, None, toy_matrix, memo)
    assert e == pytest.approx(2.0, abs=1e-9)
    assert w == 2


def test_single_solution_dead_end(toy_matrix):
    # One solution left whose member is no longer a legal guess: there is
    # nothing left to play out — dead end, whatever else is still legal.
    legal = np.ones(len(TOY), dtype=bool)
    legal[2] = False
    with pytest.raises(ValueError, match="hard-mode dead end"):
        wc.expected_and_worst_turns(np.array([2]), legal, toy_matrix, {})
    # An entirely empty legal mask is the same dead end: the member is
    # illegal, so no legal guess exists at all.
    with pytest.raises(ValueError, match="hard-mode dead end"):
        wc.expected_and_worst_turns(np.array([2]), np.zeros(len(TOY), dtype=bool), toy_matrix, {})


# --- Real word list: opening RAISE vs ABACK -------------------------------


def opening_turns(g, S_all, matrix, memo):
    """Exact turns when the opening guess is forced to index ``g``.

    The policy picks its own first guess, so a specific opening is evaluated
    by playing ``g`` and applying the pinned recursion to its buckets
    directly (the same formula expected_and_worst_turns applies one level
    down): E = 1 + sum over non-green buckets of p*E_b, W = 1 + max W_b.
    """
    patterns = matrix[g, S_all]
    counts = np.bincount(patterns, minlength=wc.N_PATTERNS)
    expected = 1.0
    worst = 0
    for pattern in np.unique(patterns):
        if pattern == wc.PATTERN_ALL_GREEN:
            continue  # solved this turn: the leading 1 covers it
        bucket = S_all[patterns == pattern]
        e_b, w_b = wc.expected_and_worst_turns(bucket, None, matrix, memo)
        expected += (counts[pattern] / S_all.size) * e_b
        worst = max(worst, w_b)
    return expected, worst + 1


@pytest.fixture(scope="module")
def real_matrix():
    words = wc.build_words()
    return words, wc.build_feedback_matrix(words)


def test_opening_matches_full_wordlist(real_matrix):
    # One shared matrix build for both openings, and one shared memo. Both
    # counts are sanity-only (not equality-pinned): distinct floats > 1,
    # with RAISE — a strong opener — expected cheaper than ABACK.
    words, matrix = real_matrix
    S_all = np.arange(len(words))
    memo = {}
    e_raise, w_raise = opening_turns(words.index("RAISE"), S_all, matrix, memo)
    e_aback, w_aback = opening_turns(words.index("ABACK"), S_all, matrix, memo)
    assert e_raise > 1
    assert e_aback > 1
    assert e_raise != e_aback
    assert e_raise < e_aback
