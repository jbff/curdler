# Wordle Co-Solver

A Python script that helps solve Wordle puzzles by suggesting the most informative word at each step using information theory principles.

## How it works

The solver uses **entropy-based optimization** to find the word that will provide the most information at each step:

1. **Information Theory**: Calculates how much each possible guess reduces the uncertainty about the solution
2. **Entropy Reduction**: Chooses the word that maximizes information gain (minimizes expected remaining solutions)
3. **Strategic Guessing**: Always suggests the word that will narrow down possibilities the most

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

### Example Session

```
🎯 Wordle Co-Solver
==================================================
Feedback format: G=Green, Y=Yellow, X=Gray
Example: GYXGY means first letter green, second yellow, third gray, etc.

Suggested starting word: RAISE

==================================================
Enter feedback (G/Y/X): xyxxx
Possible solutions remaining: 91
Next suggested word: CLOUT

==================================================
Enter feedback (G/Y/X): xyyxx
Possible solutions remaining: 8
Solutions: AGLOW, KOALA, LOAMY, LOYAL, MODAL, OFFAL, POLKA, ZONAL
Next suggested word: DOLLY
```

## Features

- **Optimal Starting Word**: Suggests the best first guess based on letter frequency and information theory
- **Adaptive Guessing**: Each subsequent guess is chosen to maximize information gain
- **Hard Mode Support**: Use `--hard` flag to enable hard mode constraints
- **Solution Tracking**: Shows remaining possible solutions (when ≤10)
- **Error Handling**: Validates feedback input and provides helpful error messages
- **Interactive**: Easy-to-use command-line interface

## Requirements

- Python 3.6+
- `wordles.txt` file with 5-letter words (one per line, uppercase)

## Algorithm Details

The solver implements:

1. **Feedback Simulation**: For each possible guess, simulates all possible feedback outcomes
2. **Entropy Calculation**: Uses Shannon entropy to measure information content
3. **Solution Filtering**: Maintains a list of words compatible with all previous feedback
4. **Information Maximization**: Always chooses the guess that provides the most expected information

This approach typically solves Wordle puzzles in 3-4 guesses on average!

## Hard Mode

Hard mode adds additional constraints:
- **Green letters** must stay in the same position in all subsequent guesses
- **Yellow letters** must be used somewhere in the word (but not in the same position)
- **Gray letters** cannot be used at all

Hard mode is more challenging but often leads to more strategic play. The solver automatically adapts its strategy when hard mode is enabled. 