"""Task 6: solver refactor on the shared core + cache-driven initial guess.

The solver fixture reuses the real wordles.txt (matrix build ~0.5 s, module
scope); cache fixtures are injected through a monkeypatched CACHE_PATH so no
test ever touches the repo root.
"""

import json
import re
import sys

import numpy as np
import pytest

import wordle_core as wc
import wordle_solver
from wordle_solver import WordleSolver, parse_feedback


@pytest.fixture(scope="module")
def solver():
    return WordleSolver("wordles.txt")


@pytest.fixture(scope="module")
def hard_solver():
    return WordleSolver("wordles.txt", hard_mode=True)


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    """Redirect the solver's cache read to a tmp file (not created unless a
    test writes it)."""
    path = tmp_path / "starting_words.json"
    monkeypatch.setattr(wordle_solver, "CACHE_PATH", str(path))
    return path


# --- Filtering equals the truth reference ---------------------------------


def test_filter_equals_truth(solver):
    # Sampled histories identical to Task 2's test_bucket_equals_legacy_filter
    # (seed 4242): the new filter's result set must equal the truth reference
    # {w in words : feedback(guess, w) == observed} — NOT the legacy word
    # filter, which is over-permissive on duplicate letters.
    n = len(solver.words)
    rng = np.random.default_rng(4242)
    for _ in range(5):
        g, w = rng.integers(0, n, size=2)
        guess = solver.words[g]
        observed = "".join(solver._get_feedback(guess, solver.words[w]))
        truth = {word for word in solver.words
                 if "".join(solver._get_feedback(guess, word)) == observed}
        solver.reset()
        solver.process_feedback(guess, list(observed))
        got = {solver.words[int(i)] for i in solver.possible_solutions}
        assert got == truth


def test_duplicate_letter_filter_fix(solver):
    # SWEEP marked XXYXX: the real rule consumes guess E's slot-by-slot, so a
    # word whose only E the true feedback greens (ABBEY: SWEEP vs ABBEY is
    # XXXGX, pattern 54) must be EXCLUDED from the XXYXX bucket. The legacy
    # filter accepted it: its yellow pass ate the word's single E, letting the
    # gray index slip past the gray check. This replacement is a deliberate
    # bug fix, not a refactor.
    pattern_xxyxx = 3**2
    assert wc.feedback_pattern("SWEEP", "ABBEY") == 2 * 27  # XXXGX, not 9
    assert wc.feedback_pattern("SWEEP", "ABBEY") != pattern_xxyxx

    solver.reset()
    solver.process_feedback("SWEEP", ["X", "X", "Y", "X", "X"])
    kept = {solver.words[int(i)] for i in solver.possible_solutions}
    truth = {word for word in solver.words
             if wc.feedback_pattern("SWEEP", word) == pattern_xxyxx}
    assert kept == truth
    assert "ABBEY" not in kept
    assert kept  # the bucket survives on the real list
    # History records pattern ints, not feedback strings.
    assert solver.guess_history == [("SWEEP", pattern_xxyxx)]


def test_process_feedback_prints_statistics(solver, capsys):
    # Pinned legacy statistics block (byte shape, values derived).
    solver.reset()
    solver.process_feedback("CRANE", ["X"] * 5)
    out = capsys.readouterr().out
    total = len(solver.possible_solutions)
    eliminated = len(solver.words) - total
    pct = f"{100 - total / len(solver.words) * 100:.1f}"
    C = wordle_solver.Colors
    assert f"\n{C.CYAN}📊 Statistics:{C.END}\n" in out
    assert f"{C.BLUE}•{C.END} Possible solutions: {C.BOLD}{total}{C.END}\n" in out
    assert f"{C.BLUE}•{C.END} Eliminated: {C.BOLD}{eliminated}{C.END} ({pct}%)\n" in out


# --- Initial guess from the analysis cache --------------------------------


def test_initial_guess_from_cache(solver, hard_solver, cache_path, capsys):
    cache_path.write_text(json.dumps({
        # Matching word count: a cache written for a different list is stale
        # and must fall back (pinned by test_initial_guess_word_count_mismatch).
        "word_count": len(solver.words),
        "best": {"expected_turns_normal": "CRANE", "hard_expected_turns": "TRAIL"}
    }))
    assert solver.get_initial_guess() == (
        "CRANE",
        pytest.approx(solver._calculate_information_gain("CRANE", solver.words)),
    )
    assert hard_solver.get_initial_guess() == (
        "TRAIL",
        pytest.approx(hard_solver._calculate_information_gain("TRAIL", hard_solver.words)),
    )
    # No fallback note when the cache is good.
    assert "default" not in capsys.readouterr().out


@pytest.mark.parametrize("corrupt", [
    None,                       # file missing entirely
    "{not json",                # unparseable
    '{"best": "CRANE"}',        # best is not a dict
    '{"best": {}}',             # metric key absent
    '{"best": {"expected_turns_normal": 3}}',  # not a string
    '{"best": {"expected_turns_normal": "ZZZZZ"}}',  # stale: not in word list
])
def test_initial_guess_cache_fallback(solver, cache_path, capsys, corrupt):
    if corrupt is not None:
        cache_path.write_text(corrupt)
    word, bits = solver.get_initial_guess()
    assert word == "SLATE"
    # Same bits the legacy entropy calculation gives for the full word list
    # (independent implementation, matching to float noise).
    legacy_bits = solver._calculate_information_gain("SLATE", solver.words)
    assert abs(bits - legacy_bits) < 1e-9
    assert bits > 0
    assert "default" in capsys.readouterr().out


def test_initial_guess_word_count_mismatch(solver, cache_path, capsys):
    # The word count must be present AND equal to the instance's list length:
    # a cache written against a different word list has list-order- specific
    # rankings, so its "best" is stale. Best word itself is valid here — only
    # the count is off — and the fallback + note must still fire.
    cache_path.write_text(json.dumps({
        "word_count": len(solver.words) + 1,
        "best": {"expected_turns_normal": "CRANE"},
    }))
    word, _ = solver.get_initial_guess()
    assert word == "SLATE"
    assert "default" in capsys.readouterr().out


# --- parse_feedback pins (existing CLI behavior) --------------------------


@pytest.mark.parametrize("bad", ["", "GYXY", "GYXX", "GYXYA"])
def test_parse_feedback_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_feedback(bad)
    # README's lowercase example stays valid.
    assert parse_feedback("xyxxx") == ["X", "Y", "X", "X", "X"]


# --- Interactive flow smoke -------------------------------------------------


def test_interactive_flow_smoke(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["wordle_solver.py"])
    answers = ["XXXXX", "quit"]

    def fake_input(prompt=""):
        return answers.pop(0) if answers else "quit"

    monkeypatch.setattr("builtins.input", fake_input)
    wordle_solver.main()
    out = capsys.readouterr().out
    assert "🚀 Suggested starting word:" in out
    assert "💡 Next suggested word:" in out
    # The suggestion line always ends with a real word, never blank (strip
    # the ANSI codes first so a colored-but-empty suggestion fails).
    line = next(line for line in out.splitlines() if "Next suggested word" in line)
    plain = re.sub(r"\033\[[0-9;]*m", "", line)
    assert bool(plain.split(":")[-1].strip())


def test_inconsistent_feedback_recovers(solver, cache_path, monkeypatch, capsys):
    # A mistyped feedback line empties S (no word can produce YGYXX after
    # CRANE). The guard must absorb it as user-input error: keep solver state
    # valid, tell the user, and let main() reoffer the previous suggestion —
    # no crash, no committed state.
    cache_path.write_text(json.dumps({
        "word_count": len(solver.words),
        "best": {"expected_turns_normal": "CRANE"},
    }))
    monkeypatch.setattr(sys, "argv", ["wordle_solver.py"])
    answers = ["YGYXX", "quit"]

    def fake_input(prompt=""):
        return answers.pop(0) if answers else "quit"

    monkeypatch.setattr("builtins.input", fake_input)

    # Capture the solver instance main() builds to assert its internal state.
    created = []
    real_cls = wordle_solver.WordleSolver

    def spy_cls(*args, **kwargs):
        instance = real_cls(*args, **kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr(wordle_solver, "WordleSolver", spy_cls)
    wordle_solver.main()  # must not raise
    out = capsys.readouterr().out
    assert "❌ No word matches that feedback — check it and try again." in out
    # The failure's statistics block showed the eliminated set (0 remaining).
    C = wordle_solver.Colors
    assert f"{C.BLUE}•{C.END} Possible solutions: {C.BOLD}0{C.END}" in out
    # The previous suggestion stands: exactly one next-suggestion line, CRANE.
    suggestion_lines = [line for line in out.splitlines() if "Next suggested word" in line]
    assert len(suggestion_lines) == 1
    assert "CRANE" in re.sub(r"\033\[[0-9;]*m", "", suggestion_lines[0])
    # Nothing of the failed entry was committed to solver state.
    instance = created[0]
    assert instance.guess_history == []
    assert instance.step_count == 0
    np.testing.assert_array_equal(
        instance.possible_solutions, np.arange(len(instance.words), dtype=np.intp)
    )
    assert instance.legal is None
