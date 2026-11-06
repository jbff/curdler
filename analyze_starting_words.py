#!/usr/bin/env python3
"""
Analyze every starting word for Wordle: normal-mode opening metrics and
exact hard-mode turn counts, all under the shipped solver policy
(``wordle_core.select_guess`` / ``expected_and_worst_turns``). Results land
in ``starting_words.json`` (repo root) for the solver to read.
"""

import datetime
import json
import math
import multiprocessing
import os
import time

import numpy as np

import wordle_core as wc
from wordle_solver import Colors

# Written next to this module (mirroring wordle_solver.CACHE_PATH), never the
# caller's cwd; for the repo module that is the repo root the solver reads.
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "starting_words.json")

NORMAL_METRICS = [
    "information_gain",
    "expected_remaining_1",
    "worst_case_1",
    "expected_turns_normal",
    "worst_turns_normal",
]
HARD_METRICS = ["hard_expected_turns", "hard_worst_turns"]
METRICS = NORMAL_METRICS + HARD_METRICS
# Fewer expected remaining is better, so these rank descending; everything
# else (all the turn/worst-bucket metrics) ranks ascending.
DESC_METRICS = {"information_gain"}  # fewer-remaining metrics rank ascending (smaller = better)
RANKING_SIZE = 100
HARD_PAIR_COUNT = 10

# Per-mode working state, filled in by ``run_analysis`` before the fork pool
# starts so every worker inherits words/matrix/counts without pickling them.
_STATE: dict = {}


def second_guess_for_largest_bucket(g: int, words: list[str], matrix: np.ndarray) -> str:
    """Best hard-mode follow-up to opening ``g``: the policy's pick for its
    largest feedback bucket, with legality narrowed to that bucket's pattern
    (hard mode only allows guesses carrying the observed information).
    """
    row = matrix[g]
    pattern = int(np.bincount(row, minlength=wc.N_PATTERNS).argmax())
    s_bucket = np.flatnonzero(row == pattern)
    legal = row == pattern
    return words[wc.select_guess(s_bucket, legal, matrix)]


def _opening_turns(
    matrix: np.ndarray, memo: dict, legal: np.ndarray | None, g: int
) -> tuple[float, int]:
    """Exact ``(expected, worst)`` turns when the opening is forced to ``g``.

    The policy picks its own first guess, so a pinned opening is evaluated one
    level down with the same recursion ``expected_and_worst_turns`` applies:
    play ``g``, solve the all-green bucket this turn, and play every other
    bucket out under the policy with the hard-mode legal mask narrowed to the
    observed pattern (``legal`` stays ``None`` in normal mode).
    """
    s_all = _STATE["s_all"]
    patterns = matrix[g][s_all]
    expected = 1.0
    worst = 0
    for pattern in np.unique(patterns):
        if pattern == wc.PATTERN_ALL_GREEN:
            continue  # solved this very turn — the leading 1 counts it
        bucket = s_all[patterns == pattern]
        child_legal = legal if legal is None else legal & (matrix[g, :] == pattern)
        e_b, w_b = wc.expected_and_worst_turns(bucket, child_legal, matrix, memo)
        expected += (bucket.size / s_all.size) * e_b
        worst = max(worst, w_b)
    return expected, worst + 1


def _normal_job(g: int) -> tuple[int, dict]:
    """Normal-mode metrics for opening ``g`` (pool worker / single path)."""
    memo = _STATE["memo"]
    entropy, expected_remaining, worst_bucket = wc.score_guess(
        _STATE["counts"][g], _STATE["s_all"].size
    )
    e_turns, w_turns = _opening_turns(_STATE["matrix"], memo, None, g)
    return g, {
        "information_gain": float(entropy),
        "expected_remaining_1": float(expected_remaining),
        "worst_case_1": int(worst_bucket),
        "expected_turns_normal": float(e_turns),
        "worst_turns_normal": int(w_turns),
    }


def _hard_job(g: int) -> tuple[int, dict]:
    """Hard-mode metrics for opening ``g`` (pool worker / single path)."""
    entry = {"hard_expected_turns": math.inf, "hard_worst_turns": math.inf}
    try:
        e_turns, w_turns = _opening_turns(
            _STATE["matrix"], _STATE["memo"], _STATE["legal"], g
        )
        entry = {"hard_expected_turns": float(e_turns), "hard_worst_turns": int(w_turns)}
    except ValueError as exc:
        if str(exc) != "hard-mode dead end":
            raise
    if _STATE["best_word_for_pair"]:
        entry["hard_pair"] = second_guess_for_largest_bucket(
            g, _STATE["words"], _STATE["matrix"]
        )
    return g, entry


def _run_jobs(job, count: int, workers: int, label: str) -> list:
    """Run ``job`` over opening indices ``0..count-1``, in order.

    ``workers <= 1`` runs single-process. Otherwise a fork pool inherits the
    already-built matrix (built before the pool starts) and ``imap`` reassembles
    results strictly in opening order — so both paths return identical results.
    """
    interval = max(1, count // 20)
    started = time.perf_counter()  # phase wall clock, shown in progress lines
    results = []
    if workers <= 1:
        for i in range(count):
            if i % interval == 0:
                _print_progress(label, i, count, started=started)
            results.append(job(i))
    else:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(workers) as pool:
            for i, result in enumerate(pool.imap(job, range(count))):
                if i % interval == 0:
                    _print_progress(label, i, count, started=started)
                results.append(result)
    _print_progress(label, count, count, done=True, started=started)
    return results


def _print_progress(
    label: str, i: int, count: int, done: bool = False, started: float | None = None
) -> None:
    if done:
        elapsed = time.perf_counter() - started if started is not None else None
        tail = f" in {elapsed:.1f}s" if elapsed is not None else ""
        print(f"{Colors.GREEN}✅ {label} complete{tail}{Colors.END}")
        return
    progress = (i / count) * 100 if count else 100.0
    tail = ""
    if started is not None and i > 0:
        elapsed = time.perf_counter() - started
        eta = elapsed / i * (count - i)  # linear projection from rate so far
        tail = f" | {elapsed:.0f}s elapsed, ~{eta:.0f}s left"
    print(f"{Colors.CYAN}{label}: {progress:.1f}% ({i}/{count}){tail}{Colors.END}")


def analyze_normal(words: list[str], matrix: np.ndarray, counts: np.ndarray, workers: int) -> dict[str, dict]:
    """Opening metrics for every word in normal mode (cheap + exact turns).

    ``counts`` is the ``counts_vs_S(matrix, all_solutions)`` pass that powers
    the cheap per-opening metrics (entropy / expected remaining / worst
    bucket); the exact turn counts reuse the policy recursion with one shared
    memo per call, and identical results whether one process runs everything
    or a fork pool splits the openings.
    """
    _STATE.update(
        words=words,
        matrix=matrix,
        counts=counts,
        s_all=np.arange(len(words), dtype=np.intp),
        memo={},  # fresh per mode call: keyed by (S bytes, legal bytes)
    )
    results = _run_jobs(_normal_job, len(words), workers, "⏳ Normal mode")
    return {words[g]: metrics for g, metrics in results}


def analyze_hard(
    words: list[str], matrix: np.ndarray, workers: int, best_word_for_pair: bool = False
) -> dict[str, dict]:
    """Exact hard-mode turn counts per opening (dead ends recorded as ``inf``).

    With ``best_word_for_pair`` each entry also carries the policy's best
    follow-up to the opening's largest bucket (the ``hard_pairs`` report data).
    """
    _STATE.update(
        words=words,
        matrix=matrix,
        s_all=np.arange(len(words), dtype=np.intp),
        legal=np.ones(len(words), dtype=bool),
        memo={},
        best_word_for_pair=best_word_for_pair,
    )
    results = _run_jobs(_hard_job, len(words), workers, "🔒 Hard mode")
    return {words[g]: metrics for g, metrics in results}


def _fmt(value: float) -> str:
    if value == math.inf:
        return "inf (dead end)"
    return f"{value:.4f}"


def _print_rankings(rankings: dict[str, list]) -> None:
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}📈 TOP 10 PER METRIC{Colors.END}")
    for metric in METRICS:
        print(f"\n{Colors.BOLD}{Colors.GREEN}{metric}{Colors.END}")
        for i, (word, value) in enumerate(rankings[metric][:10], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
            print(f"  {medal} {Colors.BOLD}{word}{Colors.END}: {Colors.BLUE}{_fmt(value)}{Colors.END}")


def _print_hard_pairs(hard: dict[str, dict], pairs: list) -> None:
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}🔒 TOP 3 HARD-MODE OPENING PAIRS{Colors.END}")
    for opening, second in pairs[:3]:
        entry = hard[opening]
        print(
            f"  {Colors.BOLD}{opening}{Colors.END} → {Colors.BOLD}{second}{Colors.END}"
            f"  (hard expected {_fmt(entry['hard_expected_turns'])},"
            f" worst {_fmt(entry['hard_worst_turns'])})"
        )


def run_analysis(words: list[str] | None = None, matrix: np.ndarray | None = None, workers: int | None = None) -> dict:
    """Analyze the full opening list and write the ``starting_words.json`` cache.

    ``words`` / ``matrix`` default to ``build_words()`` / its feedback matrix
    (tests inject toy lists); ``workers`` defaults to every core.
    """
    if words is None:
        words = wc.build_words()
    if matrix is None:
        matrix = wc.build_feedback_matrix(words)
    if workers is None:
        workers = os.cpu_count() or 1

    print(f"{Colors.BOLD}{Colors.PURPLE}🔍 Exact Starting Word Analysis{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.WHITE}Total words: {Colors.BOLD}{len(words)}{Colors.END}"
          f" | workers: {Colors.BOLD}{workers}{Colors.END}\n")

    counts = wc.counts_vs_S(matrix, np.arange(len(words), dtype=np.intp))
    started = time.perf_counter()
    normal = analyze_normal(words, matrix, counts, workers)
    normal_time = time.perf_counter() - started
    print(f"{Colors.WHITE}Normal mode took {normal_time:.1f}s{Colors.END}")
    started = time.perf_counter()
    hard = analyze_hard(words, matrix, workers, best_word_for_pair=True)
    hard_time = time.perf_counter() - started
    print(f"{Colors.WHITE}Hard mode took {hard_time:.1f}s{Colors.END}\n")

    entries = {word: {**normal[word], **hard[word]} for word in words}

    # Rankings: pinned tie-break — value order first (stable sort over the
    # alphabetical pre-sort keeps ties alphabetical), top N kept.
    rankings: dict[str, list] = {}
    for metric in METRICS:
        ranked = sorted(entries.items(), key=lambda item: item[0])
        ranked = sorted(ranked, key=lambda item: item[1][metric], reverse=metric in DESC_METRICS)
        rankings[metric] = [[word, entry[metric]] for word, entry in ranked[:RANKING_SIZE]]

    hard_pairs = [[word, hard[word]["hard_pair"]] for word, _ in rankings["hard_expected_turns"][:HARD_PAIR_COUNT]]

    cache = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "word_count": len(words),
        "best": {metric: rankings[metric][0][0] for metric in METRICS},
        "rankings": rankings,
        "hard_pairs": hard_pairs,
    }

    _print_rankings(rankings)
    _print_hard_pairs(hard, hard_pairs)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    dead_ends = sum(1 for entry in hard.values() if entry["hard_expected_turns"] == math.inf)
    print(f"\n{Colors.GREEN}💾 Cache written to starting_words.json{Colors.END}"
          f" (hard-mode dead ends: {dead_ends})")
    return cache


if __name__ == "__main__":
    run_analysis()
