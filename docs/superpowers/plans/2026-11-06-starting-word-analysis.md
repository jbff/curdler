# Starting Word Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute every definition of "optimal starting word" (normal + hard mode) exactly, every run, and make the interactive solver use the results.

**Architecture:** New shared `wordle_core.py` (numpy-vectorized feedback matrix, bucketing, guess-selection policy, expected-turns recursion) used by both the rewritten `analyze_starting_words.py` (multiprocessing driver + JSON cache) and the refactored `wordle_solver.py`. Key insight: word `w` is compatible with guess `g` + feedback `F` **iff** `matrix[g][w] == pattern(F)`, so compatibility filtering, bucketing, and hard-mode legality all collapse to matrix lookups.

**Tech Stack:** Python 3.14, numpy 2.3.5, pytest 9 (dev only). No other new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-06-starting-word-analysis-design.md` (read both; spec travels with plan).

## Global Constraints

- Word list `wordles.txt`: 2,334 unique 5-letter uppercase words; do not modify it.
- Feedback pattern encoding: integer 0–242, base-3, positional: contribution per index i is `(2*green + 1*yellow) * 3**i`; all-green = 242.
- Policy (single definition, shared by solver and analyzer): `argmax entropy; tie-break `argmin expected remaining; tie-break candidate ∈ S; tie-break list order`. If |S| == 1 and the sole candidate is legal → that word; if |S| == 1 illegal (hard-mode dead end) → best among legal.
- Every headline metric exact: no search caps, no subsampling, no scenario shortcuts.
- CLI behavior of `wordle_solver.py` (flags, prompts, output wording) unchanged.
- Existing `wordle_solver.py` public API (`WordleSolver.__init__(word_list_file, hard_mode)`, `.get_initial_guess() -> (word, float)`, `.process_feedback(guess, feedback) -> (word, float)`, `.Colors`) unchanged.
- Commit messages end with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- Tests run from repo root: `python3 -I -m pytest tests/ -v` (`-I` isolates interpreter; no pytest plugin surprises).

## Review Focus

1. Duplicate-letter feedback (e.g. guess `ERASE` vs solution `SPEED` must yield `YXXYY`, and transposed pairs `LUSHES` vs `SHELLS`-style splits) — pinned in Task 1 tests `test_duplicate_yellows` and `test_duplicate_green_exhaustion`.
2. User enters lowercase or invalid feedback in the solver → rejected with the existing error message, not a crash — Task 6 test `test_parse_feedback_rejects_invalid` (existing behavior, pinned).
3. `starting_words.json` missing or corrupt → solver falls back to labeled default `RAISE`, never crashes — Task 6 tests `test_initial_guess_cache_missing`, `test_initial_guess_cache_corrupt`.
4. Cache stale (word list edited since analysis ran; e.g. opening word no longer in list) → fallback, not a KeyError — Task 6 test `test_initial_guess_cache_stale_word`.
5. Hard-mode dead end under the policy (a remaining solution illegal for every legal guess) → `ValueError("hard-mode dead end")` surfaced, analyzer records `math.inf` — Task 4 test `test_single_solution_dead_end` (direct) and Task 5 test `test_dead_end_is_inf` (propagation).

---

### Task 1: `wordle_core.py` — feedback pattern + matrix

**Files:**
- Create: `wordle_core.py`
- Create: `tests/test_wordle_core.py`

**Interfaces:**
- Consumes: `wordles.txt`
- Produces (used by every later task):
  - `build_words(path: str = "wordles.txt") -> list[str]` (uppercase, preserves file order)
  - `feedback_pattern(guess: str, solution: str) -> int` (reference scalar implementation)
  - `PATTERN_ALL_GREEN = 242`, `N_PATTERNS = 243`
  - `build_feedback_matrix(words: Sequence[str]) -> np.ndarray[(N, N), uint8]`, `matrix[g][w] == feedback_pattern(guess_words[g], solution_words[w])`

- [ ] **Step 1: Write the failing tests** for `feedback_pattern` and the matrix. Assertions must include the exact integers below.

```python
# pattern = sum((2*g + y) * 3**i)

def test_duplicate_yellows():
    # guess ERASE vs solution SPEED -> Y X X Y Y -> digits [1,0,0,1,1]
    assert feedback_pattern("ERASE", "SPEED") == 1 + 27 + 81  # 109

def test_duplicate_green_exhaustion():
    # repeated guess letter appears once in solution (extra occurrences gray)
    p = feedback_pattern("SSSSS", "SPEED")
    # greens at i=4 (S==D? no) -> compute: only S at index 4? solution S,P,E,E,D has S once (i=0 of solution);
    # guess S's: none green (S at guess pos 0..4; solution pos0=S -> guess pos 0 is S? guess "SSSSS" pos0 S vs solution pos0 S -> green!
    # expected: i0 green (8*? 2*3**0=2), i1..i3 gray, i4 gray -> 2
    assert p == 2  # verify by hand before committing: pattern digits [2,0,0,0,0]
```

(Executor: recompute expected integers by hand for every test assertion before running; the comments show the reasoning, the integer is what's pinned.)

Also add:
- `test_simple_green_yellow` — e.g. guess `CRANE` vs `NACHO` recomputed by hand.
- `test_random_matrix_rows_match_scalar` — build the real matrix over `build_words()`, then for 20 random (g, w) pairs assert `matrix[g][w] == feedback_pattern(words[g], words[w])` and dtype uint8, shape (2334, 2334).
- `test_matrix_symmetry_of_values_not_required` — assert `matrix[g][w]` may differ from `matrix[w][g]` (guards against an accidental transpose).

- [ ] **Step 2: Run tests, expect failures** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** in `wordle_core.py`:
  - `feedback_pattern`: two-pass (greens first consuming solution slots, then yellows consuming remaining slots) — same algorithm as the existing `WordleSolver._get_feedback`.
  - `build_feedback_matrix`: fully vectorized. Encode words as (N,5) int8 letter codes; `eq[a, b, i] = solution_codes[b, i] == guess_codes[a, i]`; greens from `eq`; then loop i = 0..4 maintaining remaining-per-letter counts (N,N) excluding greens, assigning yellow where letter still available; pattern = Σ(2·green + 1·yellow)·3^i. Assert final dtype uint8 and max value ≤ 242.

- [ ] **Step 4: Run tests, expect pass.** Also run a one-off timing check of `build_feedback_matrix(build_words())` — must be < 60 s; if slower, revisit vectorization before proceeding.

- [ ] **Step 5: Commit** `feat: vectorized feedback pattern + matrix core`.

### Task 2: Bucket counts + cheap metrics

**Files:**
- Modify: `wordle_core.py`
- Test: `tests/test_wordle_core.py` (extend)

**Interfaces:**
- Produces:
  - `compatibility_set(S: np.ndarray[int], g: int, pattern: int, matrix) -> np.ndarray[int]` — indices w ∈ S with `matrix[g, w] == pattern` (this is the compatibility filter and bucket in one).
  - `counts_vs_S(matrix, S: np.ndarray[int]) -> np.ndarray[(N, 243), int32]` — for every candidate guess row, pattern counts against solution set S.
  - `score_guess(counts_row: np.ndarray[243], S_size: int) -> tuple[float, float, int]` — `(entropy_bits, expected_remaining, worst_bucket)`.
- Consumes: Task 1.

- [ ] **Step 1: Failing tests**
  - `test_bucket_equals_legacy_filter` — build `WordleSolver` (old code path), pick 5 random histories: assert new `compatibility_set` from matrix equals `[w for w in solver.possible_solutions if solver._is_word_compatible(w, guess, feedback)]` for every (g, F) sampled over the full word list (compare as word sets).
  - `test_entropy_matches_legacy` — for 5 random words, `entropy` from `counts_vs_S` row equals `solver._calculate_information_gain(word, solver.words)` within 1e-9.
  - `test_score_guess_values` — toy 3-word list `["AAAAA","BBBBB","CCCCC"]`: for guess `"AAAAA"` the buckets are all-green {AAAAA} and all-gray {BBBBB,CCCCC}, so entropy = `-(1/3)·log2(1/3) - (2/3)·log2(2/3)` ≈ **0.918296** (assert within 1e-6), expected_remaining = **5/3** (1.666667), worst = **2**.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** `counts_vs_S`: `flat = matrix[:, S].astype(np.int32) + row_offsets * N_PATTERNS` then `np.bincount(flat.ravel(), minlength=N*N_PATTERNS).reshape(N, N_PATTERNS)`. `score_guess` from counts row (skip zero counts: p = c/len(S)).

- [ ] **Step 4: Run, expect PASS. Commit** `feat: bucket counts and cheap metrics`.

### Task 3: Policy — `select_guess`

**Files:**
- Modify: `wordle_core.py`
- Test: `tests/test_wordle_core.py` (extend)

**Interfaces:**
- Consumes: Task 2.
- Produces: `select_guess(S: np.ndarray[int], legal: np.ndarray[bool] | None, matrix) -> int` implementing the Global Constraints policy verbatim.

- [ ] **Step 1: Failing tests** with a hand-built 5-word toy list, in this order: `["AAAAA","BBBBB","CCCCC","DEEED","EABCD"]` (indices 0–4), matrix via the scalar `feedback_pattern`:
  - `test_all_legal_prefers_split` — S = all 5; over all 5 candidate guesses only `EABCD` splits every pair (buckets: gray {AAAAA}, green@2 {BBBBB}, green@3 {CCCCC}, pattern 163 vs DEEED, all-green {EABCD}) → entropy log2(5) ≈ 2.3219, unique argmax → pinned result **index 4**.
  - `test_tie_prefers_member` — S = {1 (`BBBBB`), 2 (`CCCCC`)}: guesses `BBBBB`, `CCCCC`, `EABCD` all split 1/1 (entropy 1.0, expected remaining 1.0 = tie); `AAAAA`/`DEEED` give all-gray (entropy 0). Member tie-break + list order → pinned result **index 1**. (`DEEED` must **not** tie — it's the assertion that the member branch beats the order branch with non-members present.)
  - `test_hard_mode_respects_legal` — same S = {1,2} but `legal` = all `True` except index 1: best legal is **index 2** (`CCCCC`, member; `EABCD` also 1.0 but non-member).
  - `test_single_solution_legal_returns_member` — S = {2}, index 2 legal → **2** regardless of entropy.
  - `test_single_solution_illegal_falls_back` — S = {2}, legal excludes 2 but includes 4 → **4**.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** (pure numpy over `counts_vs_S`; tie-break chain exactly as Global Constraints; `legal=None` meaning "all legal"). If no candidate is legal, raise `ValueError("no legal candidates")`.

- [ ] **Step 4: Run, expect PASS. Commit** `feat: guess selection policy with pinned tie-breaks`.

### Task 4: `expected_and_worst_turns` (normal mode)

**Files:**
- Modify: `wordle_core.py`
- Test: `tests/test_turns.py` (new file)

**Interfaces:**
- Consumes: Task 3 (`select_guess`).
- Produces: `expected_and_worst_turns(S: np.ndarray[int], legal: np.ndarray[bool] | None, matrix, memo: dict, depth: int = 0) -> tuple[float, int]` — exact expectation and max depth under the policy. Memoizes on key `(S.astype(np.int32).tobytes(), legal.tobytes() if legal is not None else b"")`. Recursion rule pinned in spec/design + below.

- [ ] **Step 1: Failing tests** (toy list `["AAAAA","BBBBB","CCCCC"]`, matrix scalar-built):
  - `test_turns_two_words` — S = {0, 1}: policy picks index 0 (`AAAAA`; entropy-1.0 tie won by member + list order over the non-member perfect splitters) → buckets all-green {0} (solved this turn) and all-gray {1}; E = 1 + ½·E({1}) = **1.5**, worst = **2**.
  - `test_turns_three_words_perfect_split` — S = {0,1,2}: policy picks the non-member perfect splitter that distinguishes all three (executor: compute via `select_guess`, must split 1/1/1) → E = **2.0**, worst = **2**.
  - `test_opening_matches_full_wordlist` — on the real word list: `expected_and_worst_turns` for opening `RAISE` and for `ABACK` are distinct floats > 1 and RAISE's E is the smaller (sanity, not equality-pinned). (Slowest test in the suite — full matrix build + one shallow tree; acceptable.)
  - `test_single_solution_dead_end` — |S| == 1 with the member illegal → `pytest.raises(ValueError, match="hard-mode dead end")`; same call with an **empty** `legal` mask → `ValueError` as well.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement:**
  - |S| == 1 and member legal → (1.0, 1). Illegal → raise `ValueError("hard-mode dead end")` immediately (no guessing games).
  - Otherwise: `g = select_guess(...)`; group S by `matrix[g, S]` buckets; all-green bucket contributes exactly the leading-turn 1 (it is solved this turn); for every other bucket recurse with `legal` narrowed to `legal & (matrix[g, :] == F)` in hard mode (pass through unchanged when `legal is None`); `E = 1 + Σ buckets≠green p_b·E_b`; `W = 1 + max(W_b over non-green buckets, 0)`.
  - Depth guard: `depth > 12 → AssertionError` (policy bug detector, not a data path).

- [ ] **Step 4: Run, expect PASS. Commit** `feat: exact expected/worst turns under policy`.

### Task 5: Analyzer rewrite + JSON cache + multiprocessing

**Files:**
- Rewrite: `analyze_starting_words.py`
- Test: `tests/test_analyzer.py` (new)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `analyze_normal(words, matrix, counts, workers: int) -> dict[str, dict]` — keys per word: `information_gain, expected_remaining_1, worst_case_1, expected_turns_normal, worst_turns_normal`.
  - `analyze_hard(words, matrix, workers: int, best_word_for_pair: bool = False) -> dict[str, dict]` — per opening: `hard_expected_turns, hard_worst_turns`; plus module function `second_guess_for_largest_bucket(g, words, matrix) -> str` for the `hard_pairs` report field.
  - `run_analysis() -> dict` producing exactly this JSON cache schema (written to `starting_words.json` in repo root):
    ```json
    {"generated": "<iso8601>", "word_count": 2334,
     "best": {"information_gain": "W", "expected_remaining_1": "W", "worst_case_1": "W",
              "expected_turns_normal": "W", "worst_turns_normal": "W",
              "hard_expected_turns": "W", "hard_worst_turns": "W"},
     "rankings": {"<metric>": [["WORD", value], ... top 100], "worst_turns_normal": [...], "hard_worst_turns": [...]},
     "hard_pairs": [["OPEN", "SECOND"], ... top 10 by hard_expected_turns]}
    ```
  - Rankings tie-break: value asc (for turns-type metrics) / desc (for information_gain, expected_remaining_1 as "fewer better"; worst_case_1 asc), then alphabetical word. `best.X` = first entry of `rankings.X`.
- Colors/emoji/printing: import `Colors` from `wordle_solver` (unchanged import) for progress output; keep the progress-print style.
- The deleted code: `analyze_hard_mode_performance` and its 8 hand-picked scenarios — gone, not ported. No CLI arguments.

- [ ] **Step 1: Failing tests**
  - `test_cache_schema` — monkeypatch `build_words` to return the 3-word toy list (and stub matrix via scalar builder); `run_analysis()` returns dict with all keys above, rankings lists sorted by the pinned tie-break, `best.X == rankings.X[0][0]`.
  - `test_dead_end_is_inf` — fixture forcing a hard-mode dead end: recorded as `math.inf`, never crashes `run_analysis`.
  - `test_worker_split_matches_single` — `analyze_normal(..., workers=1) == analyze_normal(..., workers=2)` on the toy list (multiprocessing results reassembled in order, forked matrix inherited).

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** Driver: `multiprocessing.get_context("fork").Pool` over opening indices, `workers = os.cpu_count() or 1`; matrix built once before fork; progress printed as today; final section prints top-10 tables per metric (normal: entropy, expected_remaining_1, worst_case_1, expected_turns; hard: hard_expected_turns) and top-3 `hard_pairs`; then writes the cache.

- [ ] **Step 4: Run, expect PASS. Commit** `feat: exact analyzer, rankings, JSON cache`.

- [ ] **Step 5: Full run** `python3 -I analyze_starting_words.py` (expect ~2–15 min; progress visible). Record the real numbers for Task 7. Confirm `starting_words.json` written and well-formed (`python3 -I -c "import json; json.load(open('starting_words.json'))"`).

### Task 6: Solver refactor on core + cache-driven initial guess

**Files:**
- Modify: `wordle_solver.py`
- Test: `tests/test_solver.py` (new)

**Interfaces:**
- Consumes: Tasks 1–3, Task 5 cache schema.
- Produces: unchanged public API (Global Constraints). `WordleSolver` internals now: words, `matrix` (built in `__init__`), `possible_solutions` as `np.ndarray[int]`, `legal` mask (hard mode), `guess_history: list[tuple[str, int]]` (pattern ints, internal only). `get_initial_guess()` reads cache; `_filter_solutions` and next-guess search via core functions; hard-code `RAISE` only as labeled fallback.

- [ ] **Step 1: Failing tests**
  - `test_filter_equals_legacy` — construct old-style (feedback, guess) histories identical to Task 2's sampled cases; new filter result sets equal old `_filter_solutions` sets.
  - `test_initial_guess_from_cache` — write `starting_words.json` with `"best": {"expected_turns_normal": "CRANE", "hard_expected_turns": "TRAIL"}`; assert normal → `("CRANE", …)` and hard mode → `("TRAIL", …)`, both info-gain floats > 0 (computed vs full list).
  - `test_initial_guess_cache_missing`, `test_initial_guess_cache_corrupt`, `test_initial_guess_cache_stale_word` — corrupt: `"{not json"`; stale: best word not in word list; all → fallback `RAISE` and stdout (capsys) contains the literal `default` note.
  - `test_parse_feedback_rejects_invalid` — `""`, `"GYXY"`, `"GYXX"` and `"GYXYA"` variants → `ValueError` (pins existing CLI behavior; note: "XYXXX" lowercase input from README example must still work → `parse_feedback("xyxxx")` valid).
  - `test_interactive_flow_smoke` — scripted monkeypatched `input` driving one full normal-mode exchange (initial guess → feedback `XXXXX` → suggestion) does not raise and ends with a non-empty suggestion.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** Delete `allow_solutions` param (dead code), delete `_is_word_compatible`/`_is_hard_mode_compatible`'s per-word Python loops in favor of matrix lookups; `Colors` class stays (analyzer imports it). `process_feedback` output wording byte-identical. Note: `WordleSolver.__init__` builds the matrix (~seconds) — acceptable; print one line `Building feedback matrix…` only if it can't be kept silent otherwise (builder's call, prefer silent).

- [ ] **Step 4: Run, expect PASS. Commit** `feat: solver on shared core, cache-driven opening`.

### Task 7: README + final review

**Files:**
- Modify: `README.md` (analysis section, Algorithm Details, Requirements)

- [ ] **Step 1: Update README.** Replace the "Comprehensive Starting Word Analysis" section with the real Task 5 numbers (top-5 tables per metric + a one-line note that the full ranking is in `starting_words.json`), update "Algorithm Details" (policy + metric list, exact policy tie-breaks), "Requirements" line gains numpy. Keep the example session text (it may disagree with new suggestion words — replace `RAISE`/`CLOUT` snippets if the new run's actual words differ, using the real solver's words for that exact exchange only if cheap to re-derive; otherwise keep wording generic).
- [ ] **Step 2: Full manual check** `python3 -I wordle_solver.py` and `python3 -I wordle_solver.py --hard`: first suggestion matches `best.expected_turns_normal` / `best.hard_expected_turns` from Task 5's run; a couple of interactions behave as README example.
- [ ] **Step 3: Run the whole suite** `python3 -I -m pytest tests/ -v` — all pass.
- [ ] **Step 4: Commit** `docs: README analysis results + updated algorithm notes`.

## Self-review notes

- Spec → task mapping: metrics 1–3 = Task 2; metric 4–5 = Task 4; metric 6 = Tasks 4+5; policy consistency = Tasks 3+6 (one code path); cache = Task 5/6; README = Task 7; review findings 1–5 covered per Task 5 deletions + Task 6.
- Pinned-but-suspicious integers in Task 1 tests (e.g. `ERASE`/`SPEED` = 109) — executor must recompute by hand; if a pinned integer differs, the hand computation wins over the comment.
