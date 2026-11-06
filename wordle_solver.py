#!/usr/bin/env python3
"""
Wordle Co-Solver

This script helps solve Wordle puzzles by suggesting the most informative word
at each step based on information theory principles.
"""

import math
import argparse
import json
import os
from typing import List, Tuple
from collections import Counter

import numpy as np

import wordle_core as wc

_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
# Read at get_initial_guess() call time (module global, so tests can point it
# at a fixture cache); relative to this module, not the caller's cwd.
CACHE_PATH = os.path.join(_CACHE_DIR, "starting_words.json")

# Hard-coded openings used ONLY when the analysis cache is unavailable — the
# cache (Task 5) drives the real suggestion; these are the labeled fallbacks.
DEFAULT_OPENING = "RAISE"

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class WordleSolver:
    def __init__(self, word_list_file: str = "wordles.txt", hard_mode: bool = False):
        """Initialize the solver with a word list and its feedback matrix."""
        self.words = self._load_words(word_list_file)
        self.hard_mode = hard_mode
        # Prebuilt feedback matrix: matrix[g][w] == feedback_pattern(g, w).
        # Build is silent (vectorized, ~0.5 s on the real list).
        self.matrix = wc.build_feedback_matrix(self.words)
        # Possible solutions and (hard mode) legal guesses are index arrays
        # into self.words. legal is None in normal mode ("all legal"), and the
        # boolean all-mask in hard mode; both narrow by matrix lookup below.
        self.possible_solutions = np.arange(len(self.words), dtype=np.intp)
        self.legal = np.ones(len(self.words), dtype=bool) if hard_mode else None
        # (guess, pattern-int) pairs; pattern encoding per wordle_core. Internal
        # state only — the per-history legal mask is derived directly, this
        # stays for inspection/debugging.
        self.guess_history: List[Tuple[str, int]] = []
        self.step_count = 0

    def _load_words(self, filename: str) -> List[str]:
        """Load words from file."""
        with open(filename, 'r') as f:
            return [line.strip().upper() for line in f if line.strip()]

    def _get_feedback(self, guess: str, solution: str) -> List[str]:
        """Get feedback for a guess against a solution.

        Kept verbatim from the shipped solver: it is the independent
        reference implementation the pinned core tests check the matrix
        against; production filtering goes through the matrix instead.
        """
        feedback = [''] * 5
        solution_letters = list(solution)

        # First pass: mark greens
        for i in range(5):
            if guess[i] == solution[i]:
                feedback[i] = 'G'
                solution_letters[i] = None

        # Second pass: mark yellows and grays
        for i in range(5):
            if feedback[i] == '':  # Not green
                if guess[i] in solution_letters:
                    feedback[i] = 'Y'
                    # Remove the first occurrence of this letter
                    solution_letters[solution_letters.index(guess[i])] = None
                else:
                    feedback[i] = 'X'

        return feedback

    def _calculate_information_gain(self, guess: str, possible_solutions: List[str]) -> float:
        """Calculate the expected information gain of a guess.

        Kept verbatim from the shipped solver as an independent reference
        implementation (pinned by the core test suite); production scoring
        goes through wc.score_guess instead.
        """
        if not possible_solutions:
            return 0.0

        # Count how many solutions would remain for each possible feedback
        feedback_counts = Counter()

        for solution in possible_solutions:
            feedback = self._get_feedback(guess, solution)
            feedback_str = ''.join(feedback)
            feedback_counts[feedback_str] += 1

        # Calculate entropy reduction
        total_solutions = len(possible_solutions)
        information_gain = 0.0

        for count in feedback_counts.values():
            if count > 0:
                probability = count / total_solutions
                information_gain -= probability * math.log2(probability)

        return information_gain

    def _full_list_bits(self, word: str) -> float:
        """Information gain (bits) of ``word`` played against the FULL word
        list — the shape get_initial_guess() has always reported."""
        row = np.bincount(self.matrix[self.words.index(word)], minlength=wc.N_PATTERNS)
        return wc.score_guess(row, len(self.words))[0]

    def _cached_opening(self) -> str | None:
        """Best opening word from the analysis cache, or None when unusable.

        The cache is the analyzer's starting_words.json; only the opening
        metric this instance plays under is read. Any failure to parse, a
        missing/best-of-wrong-shape entry, a word count not matching this
        instance's list (a cache written against a different word list —
        ranking order is list-specific), or a stale word (not in the word
        list) means no cache, never a guessed suggestion.
        """
        metric = "hard_expected_turns" if self.hard_mode else "expected_turns_normal"
        try:
            with open(CACHE_PATH, "r") as f:
                cache = json.load(f)
        except (OSError, ValueError):  # missing file / corrupted JSON
            return None
        best = cache.get("best") if isinstance(cache, dict) else None
        if not isinstance(best, dict):
            return None
        word_count = cache.get("word_count")
        if (not isinstance(word_count, int) or isinstance(word_count, bool)
                or word_count != len(self.words)):
            return None
        word = best.get(metric)
        if isinstance(word, str) and word in self.words:
            return word
        return None

    def get_initial_guess(self) -> Tuple[str, float]:
        """Get the best starting word.

        Driven by the analysis cache (starting_words.json in the repo root);
        when it is missing, corrupted, or stale, fall back to the labeled
        default opening — the analysis is NOT auto-run (it takes minutes).
        """
        cached = self._cached_opening()
        if cached is not None:
            return cached, self._full_list_bits(cached)
        print(f"{Colors.YELLOW}ℹ️  Analysis cache unavailable (starting_words.json) — using the default opening: {DEFAULT_OPENING}{Colors.END}")
        return DEFAULT_OPENING, self._full_list_bits(DEFAULT_OPENING)

    def process_feedback(self, guess: str, feedback: List[str]) -> Tuple[str, float]:
        """Process feedback and return the best next guess."""
        g_idx = self.words.index(guess)
        # Encode the observed feedback as a pattern int (digit 2 = green,
        # 1 = yellow, 0 = gray per position). Equal to
        # feedback_pattern(guess, true_solution) because the true solution
        # lives in this very bucket, so the encoding matches the matrix.
        pattern = sum((2 if f == 'G' else 1 if f == 'Y' else 0) * 3**i
                      for i, f in enumerate(feedback))
        # Snapshot for the inconsistent-feedback guard below: on empty S the
        # state is restored to exactly this, so the caller re-enters with the
        # pre-entry state intact.
        prev_solutions = self.possible_solutions
        prev_legal = self.legal
        self.guess_history.append((guess, pattern))
        self.step_count += 1

        # Truth filter: w remains possible iff ITS OWN feedback for this guess
        # equals the observed pattern — one matrix lookup. This replaces the
        # legacy per-word filter deliberately: the legacy version is
        # over-permissive when the guess repeats a letter the feedback marks
        # yellow at one index and gray at another (its yellow pass consumes
        # the word's only occurrence, so the gray check passes a position the
        # true rule would green or yellow again). ~1 in 15 duplicate-letter
        # histories went subtly wrong under the old filter.
        self.possible_solutions = wc.compatibility_set(
            self.possible_solutions, g_idx, pattern, self.matrix
        )
        # Hard mode: legality narrows the same way (the played guess's own
        # information is all that may be reused). Normal mode keeps None.
        if self.legal is not None:
            self.legal = self.legal & (self.matrix[g_idx] == pattern)

        # Calculate statistics
        total_solutions = self.possible_solutions.size
        percentage_remaining = (total_solutions / len(self.words)) * 100

        print(f"\n{Colors.CYAN}📊 Statistics:{Colors.END}")
        print(f"   {Colors.BLUE}•{Colors.END} Possible solutions: {Colors.BOLD}{total_solutions}{Colors.END}")
        print(f"   {Colors.BLUE}•{Colors.END} Eliminated: {Colors.BOLD}{len(self.words) - total_solutions}{Colors.END} ({100 - percentage_remaining:.1f}%)")

        if total_solutions > 0 and total_solutions <= 10:
            remaining = ', '.join(self.words[int(i)] for i in self.possible_solutions)
            print(f"   {Colors.BLUE}•{Colors.END} Solutions: {Colors.GREEN}{remaining}{Colors.END}")

        # If we have one solution left, suggest it
        if total_solutions == 1:
            return self.words[int(self.possible_solutions[0])], 0.0

        # Inconsistent feedback (no word can produce it — a mistyped line):
        # a user-input error, recoverable. Undo the entry entirely so the
        # solver state stays valid for the re-entry, tell the user, and
        # return the None sentinel; main() reoffers the previous suggestion
        # and reprompts for feedback. Nothing invalid enters solver state.
        if total_solutions == 0:
            self.possible_solutions = prev_solutions
            self.legal = prev_legal
            self.guess_history.pop()
            self.step_count -= 1
            print(f"{Colors.RED}❌ No word matches that feedback — check it and try again.{Colors.END}")
            return None, 0.0

        # Policy pick (shared core): entropy/expected-remaining tie-breaks over
        # the matrix, with hard-mode legality applied. The guard above means
        # select_guess only ever sees a non-empty S here (inconsistent
        # feedback is absorbed before this line); with legality narrowed by
        # the same matrix predicates as S, the surviving solution is itself
        # legal, so S ⊆ legal — a dead end needs inconsistent input, which
        # never reaches this point.
        next_idx = wc.select_guess(self.possible_solutions, self.legal, self.matrix)

        # Report the suggested word's own information gain against the
        # current possibilities (the historical output shape).
        counts_row = np.bincount(
            self.matrix[next_idx][self.possible_solutions], minlength=wc.N_PATTERNS
        )
        entropy, _, _ = wc.score_guess(counts_row, total_solutions)
        return self.words[next_idx], entropy

    def reset(self):
        """Reset the solver for a new puzzle."""
        self.possible_solutions = np.arange(len(self.words), dtype=np.intp)
        self.legal = np.ones(len(self.words), dtype=bool) if self.hard_mode else None
        self.guess_history = []
        self.step_count = 0

def parse_feedback(feedback_str: str) -> List[str]:
    """Parse feedback string into list of feedback codes."""
    feedback_map = {'g': 'G', 'y': 'Y', 'x': 'X', 'G': 'G', 'Y': 'Y', 'X': 'X'}
    feedback = []

    for char in feedback_str.upper():
        if char in feedback_map:
            feedback.append(feedback_map[char])
        else:
            raise ValueError(f"Invalid feedback character: {char}")

    if len(feedback) != 5:
        raise ValueError(f"Feedback must be exactly 5 characters, got {len(feedback)}")

    return feedback

def display_feedback_colored(feedback: List[str]) -> str:
    """Display feedback with colored squares."""
    colored = ""
    for f in feedback:
        if f == 'G':
            colored += f"{Colors.GREEN}🟩{Colors.END}"
        elif f == 'Y':
            colored += f"{Colors.YELLOW}🟨{Colors.END}"
        else:
            colored += "⬛"
    return colored

def main():
    """Main interactive loop."""
    parser = argparse.ArgumentParser(description='Wordle Co-Solver')
    parser.add_argument('--hard', '-H', action='store_true',
                       help='Enable hard mode (all guesses must use previous information)')
    args = parser.parse_args()

    solver = WordleSolver(hard_mode=args.hard)

    mode_str = "HARD MODE" if args.hard else "NORMAL MODE"
    print(f"{Colors.BOLD}{Colors.PURPLE}🎯 Wordle Co-Solver ({mode_str}){Colors.END}")
    print(f"{Colors.CYAN}{'='*50}{Colors.END}")
    print(f"{Colors.WHITE}Feedback format: G=Green, Y=Yellow, X=Gray{Colors.END}")
    print(f"{Colors.WHITE}Example: GYXGY means first letter green, second yellow, third gray, etc.{Colors.END}")
    print(f"{Colors.WHITE}Shortcuts: 'solved' = puzzle solved, 'yes' = last guess was correct{Colors.END}")
    if args.hard:
        print(f"{Colors.YELLOW}⚠️  HARD MODE: All guesses must use information from previous guesses{Colors.END}")
    print()

    # Get initial guess
    initial_guess, initial_info = solver.get_initial_guess()
    print(f"{Colors.BOLD}{Colors.GREEN}🚀 Suggested starting word: {Colors.END}{Colors.BOLD}{initial_guess}{Colors.END}")
    print(f"{Colors.BLUE}📊 Information gain: {initial_info:.3f} bits{Colors.END}")

    while True:
        print(f"\n{Colors.CYAN}{'='*50}{Colors.END}")

        # Get feedback from user
        while True:
            try:
                feedback_str = input(f"{Colors.WHITE}Enter feedback (G/Y/X) or 'solved': {Colors.END}").strip()
                if feedback_str.lower() == 'quit':
                    return
                if feedback_str.lower() == 'solved':
                    print(f"{Colors.GREEN}🎉 Congratulations! Puzzle solved in {solver.step_count + 1} steps!{Colors.END}")
                    return
                if feedback_str.lower() == 'yes':
                    print(f"{Colors.GREEN}🎉 Congratulations! Puzzle solved in {solver.step_count + 1} steps!{Colors.END}")
                    return
                feedback = parse_feedback(feedback_str)
                break
            except ValueError as e:
                print(f"{Colors.RED}❌ Error: {e}{Colors.END}")
                continue

        # Check if we solved it
        if all(f == 'G' for f in feedback):
            print(f"{Colors.GREEN}🎉 Congratulations! Puzzle solved in {solver.step_count + 1} steps!{Colors.END}")
            return

        # Get next guess
        next_guess, next_info = solver.process_feedback(initial_guess, feedback)

        if next_guess is None:
            # Inconsistent feedback: process_feedback restored its state, so
            # the same suggestion stands — reoffer it and reprompt for
            # feedback. A typo costs one line, nothing else.
            print(f"\n{Colors.BOLD}{Colors.GREEN}💡 Next suggested word: {Colors.END}{Colors.BOLD}{initial_guess}{Colors.END}")
            continue

        # Check if we have only one solution left
        if solver.possible_solutions.size == 1:
            print(f"\n{Colors.YELLOW}🎯 Only one solution remaining: {Colors.BOLD}{next_guess}{Colors.END}")
            print(f"{Colors.WHITE}Is this the correct solution? (yes/no): {Colors.END}", end="")
            response = input().strip().lower()
            if response in ['yes', 'y']:
                print(f"{Colors.GREEN}🎉 Congratulations! Puzzle solved in {solver.step_count + 1} steps!{Colors.END}")
                return
            else:
                print(f"{Colors.RED}❌ Sorry, we've run out of possible solutions. The word list may be incomplete.{Colors.END}")
                return
        else:
            print(f"\n{Colors.BOLD}{Colors.GREEN}💡 Next suggested word: {Colors.END}{Colors.BOLD}{next_guess}{Colors.END}")
            print(f"{Colors.BLUE}📊 Information gain: {next_info:.3f} bits{Colors.END}")

        # Update for next iteration
        initial_guess = next_guess

if __name__ == "__main__":
    main()
