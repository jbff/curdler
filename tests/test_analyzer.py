"""Task 5: analyzer + JSON cache tests.

Small toy word lists injected through run_analysis's ``words`` / ``matrix``
params, so no test touches the real 2,334-word run. ``run_analysis`` writes
its cache to the current directory, so every test chdirs into ``tmp_path``
to keep the repo root pristine.
"""

import datetime
import json
import math

import numpy as np
import pytest

import analyze_starting_words as analyzer
import wordle_core as wc

TOY = ["AAAAA", "BBBBB", "CCCCC"]


def toy_matrix(words=None):
    words = words or TOY
    return np.array(
        [[wc.feedback_pattern(guess, solution) for solution in words] for guess in words],
        dtype=np.uint8,
    )


@pytest.fixture
def at_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def counts():
    return wc.counts_vs_S(toy_matrix(), np.arange(len(TOY), dtype=np.intp))


def test_cache_schema(at_root, counts):
    result = analyzer.run_analysis(words=TOY, matrix=toy_matrix(), workers=1)

    metrics = [
        "information_gain", "expected_remaining_1", "worst_case_1",
        "expected_turns_normal", "worst_turns_normal",
        "hard_expected_turns", "hard_worst_turns",
    ]
    assert set(result) == {"generated", "word_count", "best", "rankings", "hard_pairs"}
    assert result["word_count"] == len(TOY)
    # generated is real ISO-8601 UTC.
    stamp = datetime.datetime.fromisoformat(result["generated"])
    assert stamp.tzinfo is not None

    assert set(result["best"]) == set(metrics)
    assert set(result["rankings"]) == set(metrics)
    for metric in metrics:
        pairs = result["rankings"][metric]
        words_ranked = [word for word, _ in pairs]
        assert len(words_ranked) == len(set(words_ranked))  # one entry per word
        # Pinned tie-break: value order (desc for fewer-better metrics, asc
        # otherwise) with alphabetical words inside ties.
        desc = metric in ("information_gain", "expected_remaining_1")
        for (w1, v1), (w2, v2) in zip(pairs, pairs[1:]):
            if v1 == v2:
                assert w1 < w2
            else:
                assert (v1 > v2) if desc else (v1 < v2)
        assert result["best"][metric] == pairs[0][0]

    assert isinstance(result["hard_pairs"], list)
    assert len(result["hard_pairs"]) <= 10
    for opening, second in result["hard_pairs"]:
        assert opening in TOY
        assert second in TOY

    # The cache on disk is well-formed and identical to the returned dict.
    with open("starting_words.json") as f:
        assert json.load(f) == result


def test_dead_end_is_inf(at_root, counts, monkeypatch):
    real = wc.expected_and_worst_turns

    def dead_end(S, legal, matrix, memo, depth=0):
        if legal is not None:  # hard mode only
            raise ValueError("hard-mode dead end")
        return real(S, legal, matrix, memo, depth)

    monkeypatch.setattr(wc, "expected_and_worst_turns", dead_end)

    result = analyzer.run_analysis(words=TOY, matrix=toy_matrix(), workers=2)

    for metric in ("hard_expected_turns", "hard_worst_turns"):
        assert all(value == math.inf for _, value in result["rankings"][metric])
        assert result["best"][metric] in TOY  # best survives, recorded not hidden
    assert result["hard_pairs"]  # pairing still computed despite the dead ends
    with open("starting_words.json") as f:  # Infinity round-trips through json
        reloaded = json.load(f)
    assert reloaded["best"]["hard_expected_turns"] == result["best"]["hard_expected_turns"]


def test_worker_split_matches_single(at_root, counts):
    single = analyzer.analyze_normal(TOY, toy_matrix(), counts, workers=1)
    split = analyzer.analyze_normal(TOY, toy_matrix(), counts, workers=2)
    assert split == single  # pool results reassembled in opening order
    assert set(single) == set(TOY)
    assert set(single["AAAAA"]) == {
        "information_gain", "expected_remaining_1", "worst_case_1",
        "expected_turns_normal", "worst_turns_normal",
    }

    # Same invariant for the hard-mode tree walk, including the pinned pair field.
    hard_single = analyzer.analyze_hard(TOY, toy_matrix(), workers=1, best_word_for_pair=True)
    hard_split = analyzer.analyze_hard(TOY, toy_matrix(), workers=2, best_word_for_pair=True)
    assert hard_split == hard_single
    assert set(hard_single["AAAAA"]) == {
        "hard_expected_turns", "hard_worst_turns", "hard_pair",
    }
