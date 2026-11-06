"""Core Wordle feedback utilities shared by the analysis and solver work.

Feedback patterns are encoded as integers: position i contributes
(2 * green + 1 * yellow) * 3**i, so there are 3**5 = 243 possible patterns
(N_PATTERNS) and the all-green pattern is 2 * (1 + 3 + 9 + 27 + 81) = 242
(PATTERN_ALL_GREEN).
"""

from collections.abc import Sequence

import numpy as np

PATTERN_ALL_GREEN = 242
N_PATTERNS = 243

# Digit values per position: green = 2, yellow = 1, gray = 0.
_DIGIT_GREEN = 2
_DIGIT_YELLOW = 1


def build_words(path: str = "wordles.txt") -> list[str]:
    """Load the answer list: uppercase, one word per line, file order kept."""
    with open(path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]


def feedback_pattern(guess: str, solution: str) -> int:
    """Encode Wordle feedback for ``guess`` vs ``solution`` as an integer.

    Two passes, mirroring Wordle's rule (same algorithm as
    ``WordleSolver._get_feedback``): greens first, each green consuming its
    solution slot; then yellows, each yellow consuming one remaining slot of
    its letter; anything left over is gray.
    """
    slots = list(solution)
    digits = [0] * 5

    # First pass: mark greens. Each green consumes its solution slot.
    for i in range(5):
        if guess[i] == solution[i]:
            digits[i] = _DIGIT_GREEN
            slots[i] = None

    # Second pass: mark yellows. Each yellow consumes a remaining slot.
    for i in range(5):
        if digits[i] == 0 and guess[i] in slots:
            digits[i] = _DIGIT_YELLOW
            slots[slots.index(guess[i])] = None

    return sum(digit * 3**i for i, digit in enumerate(digits))


def build_feedback_matrix(words: Sequence[str]) -> np.ndarray:
    """Build ``matrix[g][w] == feedback_pattern(words[g], words[w])``.

    Fully vectorized over all (guess, solution) pairs. The green pass is one
    elementwise letter comparison; the yellow pass loops over positions
    0..4, tracking how many solution slots remain claimable per letter —
    slots claimed by greens (or by an earlier yellow of the same letter) are
    no longer available, matching the scalar two-pass algorithm.

    Returns an (N, N) uint8 array; maximum pattern value is 242.
    """
    n = len(words)
    codes = np.array([[ord(c) - ord("A") for c in w] for w in words], dtype=np.int8)

    # Pairs of letter grids, each broadcast to (N, N).
    a_idx = np.arange(n)[:, None]
    b_idx = np.arange(n)[None, :]

    # green[a, b, i]: guess a and solution b share their letter at index i.
    green = codes[:, None, :] == codes[None, :, :]  # (N, N, 5) bool

    pattern = np.zeros((n, n), dtype=np.int16)

    # Green pass: digit 2 wherever the letters match.
    for i in range(5):
        pattern += (green[:, :, i].astype(np.int16) * _DIGIT_GREEN) * 3**i

    # Remaining[a, b, l]: solution b's letter-l slots still claimable by a
    # yellow. Start from the full per-letter counts of each solution; the
    # green pass removes the slots greens consumed.
    remaining = np.empty((n, n, 26), dtype=np.int8)
    counts = (codes[:, :, None] == np.arange(26)[None, None, :]).sum(axis=1)
    remaining[:] = counts[None, :, :]
    for i in range(5):
        sol_letter_i = np.broadcast_to(codes[None, :, i], (n, n))  # codes[b, i]
        remaining[a_idx, b_idx, sol_letter_i] -= green[:, :, i].astype(np.int8)

    # Yellow pass, left to right (a later yellow must not reuse a slot an
    # earlier yellow of the same letter consumed). Digit 1 where the non-
    # green guess letter still has a claimable solution slot; the yellow
    # then consumes it.
    for i in range(5):
        guess_letter_i = np.broadcast_to(codes[:, i][:, None], (n, n))  # codes[a, i]
        avail = remaining[a_idx, b_idx, guess_letter_i]  # (N, N) copy
        yellow = ~green[:, :, i] & (avail > 0)
        remaining[a_idx, b_idx, guess_letter_i] = avail - yellow.astype(np.int8)
        pattern += yellow.astype(np.int16) * (3**i)

    result = pattern.astype(np.uint8)
    assert result.dtype == np.uint8
    assert int(result.max()) <= 242
    return result


def compatibility_set(S: np.ndarray, g: int, pattern: int, matrix: np.ndarray) -> np.ndarray:
    """Filter the candidate set ``S`` down to the pattern-``pattern`` bucket.

    A candidate ``w`` is compatible with feedback ``pattern`` for guess ``g``
    iff ``matrix[g, w] == pattern``, so the compatibility filter — and with
    it every bucket — is just a lookup in the prebuilt matrix row. ``S``
    holds matrix-row indices; returns the retained elements of ``S`` in
    order, ready to pass straight back in as ``S`` on the next round.
    """
    S = np.asarray(S, dtype=np.intp)
    row = matrix[g]
    return S[row[S] == pattern]


def counts_vs_S(matrix: np.ndarray, S: np.ndarray) -> np.ndarray:
    """For every guess row, count solutions of ``S`` landing in each pattern.

    Returns an (N, N_PATTERNS) int32 array: ``counts[g, p]`` is how many
    solution indices in ``S`` give pattern ``p`` against guess row ``g``.
    Sums over each column independently of ``len(S)`` (empty ``S`` yields
    all zeros).

    The trick: one flat ``bincount`` replaces a per-guess loop. Adding each
    guess row's own index times N_PATTERNS makes the (N, len(S)) lookup
    block contain globally unique bin ids — ``row * 243 + pattern`` — so a
    single pass tallies all rows at once and a reshape hands back the
    per-row counts.
    """
    S = np.asarray(S, dtype=np.intp)
    n = matrix.shape[0]
    flat = matrix[:, S].astype(np.int64) + np.arange(n)[:, None] * N_PATTERNS
    return np.bincount(flat.ravel(), minlength=n * N_PATTERNS).reshape(n, N_PATTERNS).astype(np.int32)


def score_guess(counts_row: np.ndarray, S_size: int) -> tuple[float, float, int]:
    """Score one guess from its pattern-count row against a set of size
    ``S_size``.

    Returns ``(entropy_bits, expected_remaining, worst_bucket)``: the
    Shannon entropy of the bucket distribution, the probability-weighted
    bucket size (sum of c^2 / |S| — how many solutions a random answer
    would leave), and the largest bucket. Zero counts are skipped so
    ``log2`` never sees 0.
    """
    counts = np.asarray(counts_row)
    nonzero = counts[counts > 0].astype(np.float64)
    p = nonzero / S_size
    entropy = float(-(p * np.log2(p)).sum())
    expected_remaining = float((nonzero * nonzero).sum() / S_size)
    return entropy, expected_remaining, int(counts.max())


_POW243 = np.int64(243) ** np.arange(8, dtype=np.int64)

def _score_counts_block(counts: np.ndarray, s_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise ``(entropy, expected_remaining)`` from a float64 counts block.

    Uses ``score_guess``'s exact formulas, vectorized over rows. The float
    arithmetic is identical to the per-row path, so a row compressed into
    this block gets bit-identical metrics to scanning it in full.
    """
    log_c = np.zeros_like(counts)
    np.log2(counts, out=log_c, where=counts > 0)
    entropy = (counts * (np.log2(s_size) - log_c)).sum(axis=1) / s_size
    expected_remaining = (counts * counts).sum(axis=1) / s_size
    return entropy, expected_remaining


def select_guess(S: np.ndarray, legal: np.ndarray | None, matrix: np.ndarray) -> int:
    """Pick the best guess index for the solution set ``S`` (Task 3 policy).

    Tie-break chain, in binding order — the first rule that breaks the tie
    wins, and each stage narrows the set of survivors:

    1. argmax entropy (pattern unpredictability of the guess against ``S``),
    2. argmin expected remaining (probability-weighted bucket size),
    3. a candidate that is itself in ``S`` (the solve-out guess),
    4. list order (lowest index first).

    ``legal`` is ``None`` for "all candidates legal", otherwise a boolean
    mask over candidate indices (hard mode). Raises
    ``ValueError("no legal candidates")`` when nothing is legal. Empty
    ``S`` is out of contract and never passed.

    Edge case: when ``S`` holds a single solution, the chain above already
    reduces to that member when it is legal (all rows tie at entropy 0 /
    expected remaining 1), but we return it explicitly so the forced win
    never touches the metrics.
    """
    S = np.asarray(S, dtype=np.intp)
    if legal is None:
        mask = np.ones(matrix.shape[0], dtype=bool)
    else:
        mask = np.asarray(legal, dtype=bool)
    candidates = np.flatnonzero(mask)  # legal guess indices, ascending
    if candidates.size == 0:
        raise ValueError("no legal candidates")

    # Forced win: exactly one solution left and it is a legal guess.
    # (When it is illegal, every metric ties and the member tie-break has
    # no member to pick, so the chain falls through to list order below.)
    if S.size == 1 and mask[S[0]]:
        return int(S[0])

    # Two-solution fast path: a member guess always splits 1/1, which wins
    # the entropy argmax (1.0 beats every non-splitter) and the
    # expected-remaining argmin (1.0 ties among splitters), then wins the
    # member stage — so the policy's pick is provably the lowest legal
    # member, without scanning any candidate rows.
    if S.size == 2 and (mask[S[0]] or mask[S[1]]):
        return int(S[0]) if mask[S[0]] else int(S[1])

    s_size = S.size
    sub = matrix[:, S]  # (L, len(S)) uint8 pattern block

    # Small solution sets: identical pattern rows give identical metrics, so
    # collapse candidates onto their distinct pattern rows first. Rows with
    # the same key share one histogram — computed once — and the scatter
    # back is exact (same float64 formula on the same histogram), so the
    # policy's choice is bit-identical to the full scan.
    if s_size <= _POW243.size and candidates.size > 3:
        keys = sub.astype(np.int64) @ _POW243[:s_size]
        uk, first, inv = np.unique(keys, return_index=True, return_inverse=True)
        if uk.size * 3 < candidates.size:
            reps = np.take(sub, first, axis=0).astype(np.int64)
            flat = reps + np.arange(uk.size)[:, None] * N_PATTERNS
            counts = np.bincount(
                flat.ravel(), minlength=uk.size * N_PATTERNS
            ).reshape(uk.size, N_PATTERNS).astype(np.float64)
            entropy, expected_remaining = _score_counts_block(counts, s_size)
            entropy = entropy[inv]
            expected_remaining = expected_remaining[inv]
        else:
            entropy, expected_remaining = None, None  # collapse not worth it
    else:
        entropy, expected_remaining = None, None

    if entropy is None:
        counts = counts_vs_S(matrix, S)[candidates].astype(np.float64)  # (L, 243)
        entropy, expected_remaining = _score_counts_block(counts, s_size)

    # The chain itself: narrow the survivors at each stage. flatnonzero
    # results stay sorted, and `best` indexes into `candidates`, so
    # whatever survives stage 4 keeps list order.
    best = np.flatnonzero(entropy == entropy.max())  # 1. argmax entropy
    er = expected_remaining[best]  # 2. argmin expected remaining
    best = best[er == er.min()]
    in_S = np.isin(candidates[best], S)  # 3. a candidate that is itself in S
    if in_S.any():
        best = best[in_S]
    return int(candidates[best[0]])  # 4. list order (lowest index first)


def expected_and_worst_turns(
    S: np.ndarray,
    legal: np.ndarray | None,
    matrix: np.ndarray,
    memo: dict,
    depth: int = 0,
) -> tuple[float, int]:
    """Exact expected and worst-case turn counts for the solution set ``S``
    under the ``select_guess`` policy (Task 4).

    Returns ``(expected_turns, worst_turns)`` for playing ``S`` out under the
    policy: each turn the policy's pick ``g`` is played, the all-green bucket
    of ``S`` is solved that same turn (covered by the leading +1), and every
    other bucket is played out recursively:

        E(S) = 1 + Σ_{buckets ≠ green} p_b · E(S_b)
        W(S) = 1 + max(W(S_b) over buckets ≠ green, 0)

    ``legal`` is ``None`` for normal mode ("all candidates legal") or a
    boolean mask over candidate indices (hard mode). Hard-mode legality
    narrows like the solution set: after observing pattern ``F`` against
    guess ``g``, only words compatible with that feedback remain legal
    guesses, so the child mask is ``legal & (matrix[g, :] == F)`` — the
    same matrix-row lookup the bucketing itself uses. In normal mode nothing
    narrows, so ``None`` passes through unchanged.

    Results are memoized in ``memo`` on the key
    ``(S.astype(np.int32).tobytes(), legal.tobytes() if legal is not None
    else b"")``. The legal-mask bytes matter beyond the ``S`` bytes: two
    hard-mode calls can share a candidate set but differ in which of its
    members are still legal, and those states must not collide — hence the
    empty bytes literal standing in for ``None``.

    Raises:
        ValueError("hard-mode dead end") — ``S`` is down to one solution
            that is no longer a legal guess (nothing left to play out).
        ValueError("no legal candidates") — nothing is legal; raised by
            ``select_guess`` and propagated.
        AssertionError — ``depth`` exceeds 12: a policy-bug detector, not a
            data path (a healthy policy solves any ``S`` long before this).
    """
    if depth > 12:
        raise AssertionError("policy bug: expected_and_worst_turns deeper than depth 12")

    S = np.asarray(S, dtype=np.intp)

    # Forced win: one solution left and it is still a legal guess.
    if S.size == 1 and (legal is None or np.asarray(legal, dtype=bool)[S[0]]):
        return 1.0, 1
    # One solution left but unreachable by any legal guess: hard-mode dead end.
    if S.size == 1:
        raise ValueError("hard-mode dead end")

    # Two-solution fast path (exact, not an approximation): the policy plays
    # the lowest legal member — proven by select_guess's 1/1-split argument —
    # which solves its own pattern this turn and leaves exactly one solution
    # for the next, so E = 1 + (1/2)·1 and W = 2 bit-for-bit.
    if S.size == 2 and (legal is None or np.asarray(legal, dtype=bool)[S].all()):
        return 1.5, 2

    key = (S.astype(np.int32).tobytes(), legal.tobytes() if legal is not None else b"")
    if key in memo:
        return memo[key]

    g = select_guess(S, legal, matrix)
    patterns = matrix[g][S]
    total = S.size

    expected = 1.0  # this turn: the policy's pick is played
    worst = 0  # max over non-green children, floored at 0 by the formula
    for pattern in np.unique(patterns):
        if pattern == PATTERN_ALL_GREEN:
            continue  # solved this very turn — the leading 1 already counts it
        bucket = S[patterns == pattern]
        child_legal = legal if legal is None else legal & (matrix[g, :] == pattern)
        e_b, w_b = expected_and_worst_turns(bucket, child_legal, matrix, memo, depth + 1)
        expected += (bucket.size / total) * e_b
        if w_b > worst:
            worst = w_b

    result = (expected, worst + 1)
    memo[key] = result
    return result
