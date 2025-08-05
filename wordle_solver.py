#!/usr/bin/env python3
"""
Wordle Co-Solver

This script helps solve Wordle puzzles by suggesting the most informative word
at each step based on information theory principles.
"""

import math
import argparse
from typing import List, Dict, Tuple, Set
from collections import Counter

class WordleSolver:
    def __init__(self, word_list_file: str = "wordles.txt", hard_mode: bool = False):
        """Initialize the solver with a word list."""
        self.words = self._load_words(word_list_file)
        self.possible_solutions = self.words.copy()
        self.guess_history = []
        self.hard_mode = hard_mode
        
    def _load_words(self, filename: str) -> List[str]:
        """Load words from file."""
        with open(filename, 'r') as f:
            return [line.strip().upper() for line in f if line.strip()]
    
    def _get_feedback(self, guess: str, solution: str) -> List[str]:
        """Get feedback for a guess against a solution."""
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
    
    def _is_word_compatible(self, word: str, guess: str, feedback: List[str]) -> bool:
        """Check if a word is compatible with the given guess and feedback."""
        word_letters = list(word)
        guess_letters = list(guess)
        
        # Check greens first
        for i in range(5):
            if feedback[i] == 'G':
                if word[i] != guess[i]:
                    return False
                word_letters[i] = None
                guess_letters[i] = None
        
        # Check yellows and grays
        for i in range(5):
            if feedback[i] == 'Y':
                # Letter must be in word but not at this position
                if guess[i] not in word_letters or word[i] == guess[i]:
                    return False
                # Remove the first occurrence of this letter
                if guess[i] in word_letters:
                    word_letters[word_letters.index(guess[i])] = None
            elif feedback[i] == 'X':
                # Letter must not be in word
                if guess[i] in word_letters:
                    return False
        
        return True
    
    def _is_hard_mode_compatible(self, word: str) -> bool:
        """Check if a word satisfies hard mode constraints."""
        if not self.hard_mode or not self.guess_history:
            return True
        
        for guess, feedback in self.guess_history:
            if not self._is_word_compatible(word, guess, feedback):
                return False
        
        return True
    
    def _filter_solutions(self, guess: str, feedback: List[str]) -> List[str]:
        """Filter possible solutions based on guess and feedback."""
        return [word for word in self.possible_solutions 
                if self._is_word_compatible(word, guess, feedback)]
    
    def _calculate_information_gain(self, guess: str, possible_solutions: List[str]) -> float:
        """Calculate the expected information gain of a guess."""
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
    
    def _get_best_guess(self, possible_solutions: List[str], allow_solutions: bool = True) -> str:
        """Find the word that provides the most information."""
        best_guess = None
        best_information = -1
        
        # In hard mode, we can only use words that satisfy hard mode constraints
        if self.hard_mode:
            candidate_words = [w for w in possible_solutions if self._is_hard_mode_compatible(w)]
            if not candidate_words:
                # If no solutions satisfy hard mode, we have to use any compatible word
                candidate_words = [w for w in self.words if self._is_hard_mode_compatible(w)]
        else:
            # Consider all words as potential guesses
            candidate_words = self.words if allow_solutions else [w for w in self.words if w not in possible_solutions]
        
        for guess in candidate_words:
            information = self._calculate_information_gain(guess, possible_solutions)
            if information > best_information:
                best_information = information
                best_guess = guess
        
        return best_guess
    
    def get_initial_guess(self) -> str:
        """Get the best starting word."""
        return self._get_best_guess(self.words)
    
    def process_feedback(self, guess: str, feedback: List[str]) -> str:
        """Process feedback and return the best next guess."""
        # Add to history
        self.guess_history.append((guess, feedback))
        
        # Filter possible solutions
        self.possible_solutions = self._filter_solutions(guess, feedback)
        
        print(f"Possible solutions remaining: {len(self.possible_solutions)}")
        if len(self.possible_solutions) <= 10:
            print(f"Solutions: {', '.join(self.possible_solutions)}")
        
        # If we have one solution left, suggest it
        if len(self.possible_solutions) == 1:
            return self.possible_solutions[0]
        
        # Get best next guess
        return self._get_best_guess(self.possible_solutions)
    
    def reset(self):
        """Reset the solver for a new puzzle."""
        self.possible_solutions = self.words.copy()
        self.guess_history = []

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

def main():
    """Main interactive loop."""
    parser = argparse.ArgumentParser(description='Wordle Co-Solver')
    parser.add_argument('--hard', '-H', action='store_true', 
                       help='Enable hard mode (all guesses must use previous information)')
    args = parser.parse_args()
    
    solver = WordleSolver(hard_mode=args.hard)
    
    mode_str = "HARD MODE" if args.hard else "NORMAL MODE"
    print(f"🎯 Wordle Co-Solver ({mode_str})")
    print("=" * 50)
    print("Feedback format: G=Green, Y=Yellow, X=Gray")
    print("Example: GYXGY means first letter green, second yellow, third gray, etc.")
    if args.hard:
        print("⚠️  HARD MODE: All guesses must use information from previous guesses")
    print()
    
    # Get initial guess
    initial_guess = solver.get_initial_guess()
    print(f"Suggested starting word: {initial_guess}")
    
    while True:
        print("\n" + "=" * 50)
        
        # Get feedback from user
        while True:
            try:
                feedback_str = input("Enter feedback (G/Y/X): ").strip()
                if feedback_str.lower() == 'quit':
                    return
                feedback = parse_feedback(feedback_str)
                break
            except ValueError as e:
                print(f"Error: {e}")
                continue
        
        # Check if we solved it
        if all(f == 'G' for f in feedback):
            print("🎉 Congratulations! Puzzle solved!")
            return
        
        # Get next guess
        next_guess = solver.process_feedback(initial_guess, feedback)
        print(f"Next suggested word: {next_guess}")
        
        # Update for next iteration
        initial_guess = next_guess

if __name__ == "__main__":
    main() 