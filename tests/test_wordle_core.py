import math

import numpy as np
import pytest

import wordle_core as wc
import wordle_solver

# encoding: pattern = sum((2 * green + 1 * yellow) * 3**i) — one ternary digit
# per position (2 = green, 1 = yellow, 0 = gray).


def test_duplicate_yellows():
    # guess ERASE vs solution SPEED: no greens; E (pos 0), S (pos 3), E (pos 4)
    # are yellow, R and A gray -> Y X X Y Y -> digits [1, 0, 0, 1, 1]
    assert wc.feedback_pattern("ERASE", "SPEED") == 1 + 27 + 81  # 109


def test_duplicate_green_exhaustion():
    # guess SSSSS vs solution SPEED: S == S only at index 0 -> one green;
    # the solution holds a single S, so the four extra guess S's are gray.
    # digits [2, 0, 0, 0, 0] -> 2
    assert wc.feedback_pattern("SSSSS", "SPEED") == 2


def test_simple_green_yellow():
    # guess CRANE vs solution NACHO: no greens; hand check of the yellow pass:
    # C yellow (consumes solution C), R gray, A yellow, N yellow, E gray
    # -> Y X Y Y X -> digits [1, 0, 1, 1, 0] -> 1 + 9 + 27 = 37
    assert wc.feedback_pattern("CRANE", "NACHO") == 1 + 9 + 27  # 37
    # all green: digits [2, 2, 2, 2, 2] -> 2 * (1 + 3 + 9 + 27 + 81) = 242
    assert wc.feedback_pattern("SPEED", "SPEED") == wc.PATTERN_ALL_GREEN


def test_pattern_constants():
    assert wc.PATTERN_ALL_GREEN == 242
    assert wc.N_PATTERNS == 243


def test_build_words():
    words = wc.build_words()
    assert len(words) == 2358
    assert words[0] == "ABACK"
    assert all(w == w.upper() for w in words)


@pytest.fixture(scope="module")
def real_matrix():
    words = wc.build_words()
    return words, wc.build_feedback_matrix(words)


def test_random_matrix_rows_match_scalar(real_matrix):
    words, matrix = real_matrix
    assert matrix.dtype == np.uint8
    assert matrix.shape == (2358, 2358)
    rng = np.random.default_rng(12345)
    for _ in range(20):
        g, w = rng.integers(0, len(words), size=2)
        assert matrix[g, w] == wc.feedback_pattern(words[g], words[w])


def test_matrix_matches_scalar_on_toy_list():
    toy = ["SPEED", "ERASE", "SSSSS", "CRANE", "NACHO"]
    matrix = wc.build_feedback_matrix(toy)
    for gi, guess in enumerate(toy):
        for si, solution in enumerate(toy):
            assert matrix[gi, si] == wc.feedback_pattern(guess, solution)


def test_matrix_symmetry_of_values_not_required(real_matrix):
    _words, matrix = real_matrix
    # Guards against an accidental transpose: some (g, w) pairs must differ
    # from their mirror (w, g).
    assert bool((matrix != matrix.T).any())


# --- Task 2: bucket counts + cheap metrics -------------------------------


@pytest.fixture(scope="module")
def legacy_solver():
    return wordle_solver.WordleSolver("wordles.txt")


def test_bucket_equals_legacy_filter(real_matrix, legacy_solver):
    # compatibility_set(S, g, F, matrix) must reduce to the compatibility
    # filter: w is compatible with (guess, feedback) iff feedback(guess, w)
    # equals the pattern. The reference below is built from the legacy
    # solver's own _get_feedback (pure legacy code). _is_word_compatible is
    # NOT fit as reference: it is over-permissive when the guess repeats a
    # letter that F marks yellow at one index and gray at another — its
    # yellow pass consumes the word's only occurrence of the letter, so a
    # position F marks gray escapes the gray check even where the true
    # feedback would be green (e.g. SWEEP/XXYXX: it accepts {SWEEP-real-
    # feedback = XXXGX} words like ABBEY that the real rule excludes).
    words, matrix = real_matrix
    solver = legacy_solver
    assert solver.words == words  # both loaded from the same wordles.txt
    n = len(words)
    rng = np.random.default_rng(4242)
    for _ in range(5):
        g, w = rng.integers(0, n, size=2)
        guess = words[g]
        feedback = "".join(solver._get_feedback(guess, words[w]))
        truth = {word for word in solver.words
                 if "".join(solver._get_feedback(guess, word)) == feedback}
        bucket = wc.compatibility_set(np.arange(n), g,
                                      wc.feedback_pattern(guess, words[w]),
                                      matrix)
        assert {words[i] for i in bucket} == truth


def test_entropy_matches_legacy(real_matrix, legacy_solver):
    # Shannon entropy of the pattern distribution from counts_vs_S must
    # equal the legacy _calculate_information_gain (same quantity).
    words, matrix = real_matrix
    S = np.arange(len(words))
    counts = wc.counts_vs_S(matrix, S)
    rng = np.random.default_rng(7)
    for g in rng.choice(len(words), size=5, replace=False):
        entropy, _expected_remaining, _worst = wc.score_guess(counts[g], len(S))
        want = legacy_solver._calculate_information_gain(words[g], legacy_solver.words)
        assert abs(entropy - want) < 1e-9


def test_score_guess_values():
    # Toy: guess AAAAA buckets {AAAAA} (all-green) and {BBBBB, CCCCC}
    # (all-gray). p = 1/3 and 2/3.
    toy = ["AAAAA", "BBBBB", "CCCCC"]
    matrix = wc.build_feedback_matrix(toy)
    S = np.arange(3)
    counts = wc.counts_vs_S(matrix, S)
    entropy, expected_remaining, worst = wc.score_guess(counts[0], len(S))
    want_entropy = -(1 / 3) * math.log2(1 / 3) - (2 / 3) * math.log2(2 / 3)
    assert abs(entropy - want_entropy) < 1e-6
    assert abs(entropy - 0.918296) < 1e-6
    # Sum over buckets of p * size = (1^2 + 2^2) / 3.
    assert abs(expected_remaining - 5 / 3) < 1e-6
    assert abs(expected_remaining - 1.666667) < 1e-6
    assert worst == 2
