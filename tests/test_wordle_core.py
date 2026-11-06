import numpy as np
import pytest

import wordle_core as wc

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
    assert len(words) == 2334
    assert words[0] == "ABACK"
    assert all(w == w.upper() for w in words)


@pytest.fixture(scope="module")
def real_matrix():
    words = wc.build_words()
    return words, wc.build_feedback_matrix(words)


def test_random_matrix_rows_match_scalar(real_matrix):
    words, matrix = real_matrix
    assert matrix.dtype == np.uint8
    assert matrix.shape == (2334, 2334)
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
