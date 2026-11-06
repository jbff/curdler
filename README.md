# Wordle Co-Solver

A Python script that helps solve Wordle puzzles with exact information-theoretic analysis — an expected-turns-optimal opening in normal and hard mode, recomputed from the full candidate pool.

## How it works

The solver uses **entropy-based optimization** to find the word that will provide the most information at each step:

1. **Information Theory**: Calculates how much each possible guess reduces the uncertainty about the solution
2. **Entropy Reduction**: Chooses the word that maximizes information gain (minimizes expected remaining solutions)
3. **Strategic Guessing**: Always suggests the word that will narrow down possibilities the most

## Features

- **🎯 Hard Mode Support**: Use `--hard` flag to enable hard mode constraints
- **📊 Real-time Statistics**: Shows information gain, elimination percentages, and solution counts
- **🔢 Step Counting**: Tracks and displays the number of steps taken to solve
- **⚡ Smart Shortcuts**: Type 'solved' or 'yes' to quickly indicate puzzle completion
- **💡 Solution Validation**: Handles edge cases when only one solution remains
- **🎨 Visual Feedback**: Visual feedback patterns with colored squares

## Usage

```bash
# Normal mode
python3 wordle_solver.py

# Hard mode
python3 wordle_solver.py --hard
```

### Feedback Format

Enter feedback using these codes:
- **G** = Green (letter is correct and in right position)
- **Y** = Yellow (letter is in word but wrong position)  
- **X** = Gray (letter is not in word)

Example: `GYXGY` means:
- 1st letter: Green
- 2nd letter: Yellow  
- 3rd letter: Gray
- 4th letter: Green
- 5th letter: Yellow

### Smart Shortcuts

- Type `solved` to indicate the puzzle is complete
- Type `yes` to confirm the last guess was correct
- Type `quit` to exit the program

### Example Session

```
🎯 Wordle Co-Solver (NORMAL MODE)
==================================================
Feedback format: G=Green, Y=Yellow, X=Gray
Example: GYXGY means first letter green, second yellow, third gray, etc.
Shortcuts: 'solved' = puzzle solved, 'yes' = last guess was correct

🚀 Suggested starting word: SLATE
📊 Information gain: 5.858 bits

==================================================
Enter feedback (G/Y/X) or 'solved': xxyyx

📊 Statistics:
   • Possible solutions: 52
   • Eliminated: 2307 (97.8%)

💡 Next suggested word: TAINT
📊 Information gain: 4.307 bits

==================================================
Enter feedback (G/Y/X) or 'solved': xgggg

📊 Statistics:
   • Possible solutions: 2
   • Eliminated: 2357 (99.9%)
   • Solutions: FAINT, PAINT

💡 Next suggested word: FAINT
📊 Information gain: 1.000 bits
```

## Hard Mode

Hard mode adds additional constraints:
- **Green letters** must stay in the same position in all subsequent guesses
- **Yellow letters** must be used somewhere in the word (but not in the same position)
- **Gray letters** cannot be used at all

Hard mode is more challenging but often leads to more strategic play. The solver automatically adapts its strategy when hard mode is enabled.

## Statistics Displayed

The solver provides real-time statistics including:
- **Information gain** in bits for each suggested word
- **Number of possible solutions** remaining
- **Percentage of words eliminated** from the original list
- **Step count** when puzzle is solved

## Algorithm Details

The solver and the analyzer share one core (`wordle_core.py`):

1. **Feedback matrix**: the feedback of every (guess, solution) pair is precomputed
   once as a vectorized square table (N×N over the word list) — compatibility filtering, candidate
   bucketing, and hard-mode legality are all single lookups into it.
2. **One policy, everywhere**: every suggestion (and every metric in the analysis)
   is the *exact* expected-turns tree of one pinned rule: maximize information,
   tie-break by fewer remaining candidates, then prefer a word that could still be
   the answer, then list order. The metrics describe the solver you are actually playing.
3. **Exact metrics**: expected and worst turns to solve are computed exactly for
   every opening word, in normal and hard mode (no heuristics, no subsampling).

Under this policy the expected solve length is ~3.46–3.54 guesses (normal) from
the best opening — typically 3–4 guesses as before. Word filtering also fixed a
long-standing bug the old per-word filter had with repeated letters in guesses.

**Greedy, not globally optimal:** the policy picks the best word *for the current
state* each turn. It is not a globally optimal decision tree, which could shave a
few hundredths of a guess off the average (≈3.42 is achievable with a full
look-ahead tree in normal mode). The trade is deliberate: greedy keeps every
turn's decision instant, and its expected-turns cost is exactly what the analyzer
measures — the metric and the play agree by construction.

## Requirements

- Python 3.10+
- numpy (the feedback matrix and per-turn searches are vectorized)
- `wordles.txt` file with 5-letter words (one per line, uppercase); the answer
  pool. Run `python3 analyze_starting_words.py` to (re)build `starting_words.json`
  (~10 min on 20 cores; the solver falls back to a labeled default without it).
- Terminal that supports ANSI color codes

## 🔍 Exact Starting Word Analysis

Computed by `analyze_starting_words.py` over every answer in `wordles.txt`
(exact expected-turn trees per opening, normal and hard mode; full ranking in
`starting_words.json`). Analysis takes ~8 min normal + ~3 min hard on 20 cores.

Metrics:
- **information gain** — bits of Shannon entropy removed from the candidate pool
  (one guess). Kept for comparison; it is *not* the headline metric.
- **expected / worst turns to solve** — exactly what it says, under the solver's
  policy. This is the headline metric: information is a means, turns are the score.
- **expected remaining after 1 guess** and **largest remaining bucket** — the
  single-guess split quality (smaller is better for both).

### 📈 TOP 10 BY EXPECTED TURNS (Normal Mode)

🥇 **SLATE**: 3.463 avg turns, worst 6
🥈 **CRATE**: 3.471 avg turns, worst 6
🥉 **SLANT**: 3.472 avg turns, worst 6
4. **CRANE**: 3.472 avg turns, worst 6
5. **LEAST**: 3.473 avg turns, worst 6
6. **TRACE**: 3.473 avg turns, worst 6
7. **STALE**: 3.480 avg turns, worst 6
8. **STARE**: 3.483 avg turns, worst 6
9. **SLICE**: 3.487 avg turns, worst 6
10. **LANCE**: 3.487 avg turns, worst 6

Best single-guess information: **RAISE** (5.881 bits), then SLATE (5.858).
Notably, the entropy-optimal opener is *not* the turns-optimal opener: RAISE
leaves a slightly lumpy distribution, which costs more expected turns than its
extra bits are worth.

### 🔒 TOP 10 BY EXPECTED TURNS (Hard Mode)

🥇 **SLATE**: 3.539 avg turns, worst 8  (best 2nd guess for its most likely bucket: CRONY)
🥈 **LEAST**: 3.542 avg turns, worst 8  (best 2nd guess: CRONY)
🥉 **LEANT**: 3.550 avg turns, worst 9  (best 2nd guess: CROUP)
4. **SLANT**: 3.552 avg turns, worst 8
5. **TRACE**: 3.553 avg turns, worst 8
6. **STALE**: 3.556 avg turns, worst 8
7. **CRANE**: 3.560 avg turns, worst 8
8. **CRATE**: 3.560 avg turns, worst 8
9. **TRICE**: 3.565 avg turns, worst 8
10. **REACT**: 3.566 avg turns, worst 8

### 🏆 FINAL RECOMMENDATIONS

**Use SLATE in both modes.** The margins behind it (CRANE/CRATE/TRACE, all
within 0.01 turns) are inside measurement-irrelevant territory — these five
openers are effectively interchangeable, and any of them is defensible. The old
RAISE (normal) / CANOE (hard) recommendations ranked by the wrong metrics:
entropy for normal mode, and a scenario heuristic for hard mode that measured
nothing about solving.

CANOE's actual hard-mode cost under the exact metric: 3.608 avg turns —
ranked outside the top 100. Against the NYTimes WordleBot it can still look
strong, because their one-step scorer is weaker in hard mode than any
exact-turns policy.

### 🌳 Word list genealogy

`wordles.txt` doubles as the solution and guess pool. Its provenance:

1. The original Wordle source code (extracted Jan 2022) defines an **answer
   list of 2,315** words scheduled through ~2027 and a much larger allowed-guess
   pool (~12,972+, later expanded by NYT toward ~14,855).
2. The NYTimes removed six answer words (AGORA, FIBRE, LYNCH, PUPAL, SLAVE,
   WENCH) and stopped the static list in 2022-11 — answers are now chosen
   per-day, so **no complete official list exists after that date**.
3. This list = the NYT-era snapshot (2,309 words) plus **individually verified
   later daily answers** (2024–2026 picks like GRIFT, GUNKY, PSHAW), each checked
   against independent answer archives before joining. Guess-only words
   (e.g. INSET, MUTED) are deliberately excluded — better too few valid
   solutions than silent false positives.

Because answers can now repeat (NYT began deliberately reusing words in 2026),
the list only ever grows. After expanding it, re-run the analyzer (~11 min) —
the solver detects a stale cache by word count and falls back to a labeled
default opening otherwise.

If guess-only words become interesting for the *guess* side (pure-information
openers), that wants a second list — the official allowed-guesses pool —
plumbed in as a separate candidate source; the matrix is already shaped for it.

### ⚙️ How the analyzer computes this

`wordle_core.py` precomputes the feedback of every (guess, answer) pair once —
a vectorized N×N table over the word list (a few MB, ~1s to build). After that, every metric
is arithmetic on that table:

- Compatibility with any feedback pattern is a single lookup: word *w* survives
  feedback *F* on guess *g* exactly when `matrix[g][w] == pattern(F)` — the same
  lookup buckets answers and enforces hard-mode legality.
- Per opening word, answers are grouped by feedback pattern ("buckets"); the
  exact expected-turns tree recurses over buckets, choosing each turn's guess
  with the shared policy, memoizing by state so shared subtrees solve once.
- Deep-tree nodes with few remaining answers shortcut analytically (e.g. two
  answers left is provably 1.5/2.0 turns); remaining small nodes compress
  identical candidate rows first. Results are bit-identical to the plain slow
  implementation — 10× faster.
- The per-opening work runs under a 20-process fork pool; results reassemble
  in opening order. Output: the tables above plus `starting_words.json`
  (top-100 ranking per metric, the best hard-mode pair per opener).
