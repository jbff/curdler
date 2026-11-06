# Starting Word Analysis — Design

## Intent

The repo's stated goal is to find the most optimal opening words for Wordle
(normal and hard mode). Today the "optimization" is hard-coded in
`wordle_solver.py` (RAISE / CANOE) and the analysis script that is supposed to
back it up uses a metric that does not measure solving success (count of
remaining hard-mode-legal guess candidates after 8 hand-picked scenarios).
This change makes the analysis compute, exactly, every meaningful definition
of "optimal", every run, and fixes the review findings.

## Goals

- Compute, for every word in `wordles.txt`, all of: information gain (entropy),
  expected remaining candidates after 1 guess, worst-case remaining bucket,
  expected/worst turns to solve under a defined continuation policy.
- Compute hard-mode optimal **opening pairs** (1st guess + best 2nd guess per
  feedback bucket), expected and worst turns.
- Every metric exact (no search caps, no approximations, no hand-picked
  scenarios).
- Rank and display top words per metric; write results to a cache file.
- Fix review findings (below).

## Non-goals

- No changes to `wordles.txt`.
- No second-guess pair optimization for normal mode (opening pair only for
  hard mode, where the constraint makes the pair the meaningful unit).
- No change to the interactive CLI flow beyond bug fixes and speed.

## Anti-goals

- Any silently approximated headline metric.

## Approaches considered

- **A. Exact metrics, numpy-vectorized + multiprocessing** (chosen): all
  metrics exact, full run a few minutes on 20 cores.
- B. Pure-Python exact + multiprocessing: same results, 30–90 min per run.
- C. Cheap pass for all + exact deep pass for top ~100: fast but headline
  expected-turns ranking can in principle miss, and output must caveat it.

User picked A (2026-11-06) because the analysis is rerun occasionally, not
constantly, and the extra minutes buy exactness.

## Metric definitions

Bucket = group of answer-list words giving identical feedback to a guess.

1. `information_gain` — Shannon entropy (bits) of the feedback distribution
   over the answer list. (What the current code does; kept, not headline.)
2. `expected_remaining_1` — E[bucket size] after the opening guess.
3. `worst_case_1` — size of the largest bucket.
4. `expected_turns_normal` — expected number of guesses to solve, exactly,
   under the greedy policy: each turn guess = argmax entropy over the word
   list (tie-break: lower expected remaining, then word-list order), with
   1-turn credit when a remaining solution is among the ties/unique.
5. `worst_turns_normal` — maximum depth of that decision tree.
6. `hard_expected_turns` / `hard_worst_turns` — same policy but each guess
   after the first must satisfy hard-mode compatibility with all previous
   feedback; ranked by opening word (the pair is internal to the policy).

Policy consistency requirement: the analyzer's continuation policy must be
the same code as the solver's suggestion policy, or the metrics would
describe a solver nobody is playing.

## Implementation (what's left to the builder)

- New shared core module: feedback generation, bucketing, policy; numpy
  vectorized (precomputed 2334×2334 uint8 feedback matrix, 5.4 MB; vectorized
  per-candidate count via flat bincount), multiprocessing across 20 cores.
- Rewrite `analyze_starting_words.py` on top of the core; print rankings for
  every metric; write `starting_words.json` cache (per-metric ranking lists).
- Fix `wordle_solver.py`:
  - `get_initial_guess()` reads the cache (hard-coded values only as a
    clearly-labeled fallback when cache is absent).
  - Bug fixes from review. Keep CLI behavior; faster turn response via core.
- Update README analysis section with new output.

## Review findings being fixed

1. Starting words hard-coded, not computed (`wordle_solver.py:156-168`).
2. Hard-mode "score" metric doesn't measure solving (replaced entirely)
   (`analyze_starting_words.py:53-86`).
3. Entropy-only ranking ignores 1-guess wins → turns-based metrics added.
4. No caching of feedback computations → precomputed matrix.
5. Analyzer recomputes information gain it already had; 18,672 temp solvers
   constructed (8 per word) → removed.
