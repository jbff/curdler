# Wordle Co-Solver

A Python script that helps solve Wordle puzzles by suggesting the most informative word at each step using information theory principles.

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

🚀 Suggested starting word: RAISE
📊 Information gain: 5.878 bits

==================================================
Enter feedback (G/Y/X) or 'solved': xyxxx

📊 Statistics:
   • Possible solutions: 91
   • Eliminated: 2218 (96.1%)
   • Solutions: AGLOW, KOALA, LOAMY, LOYAL, MODAL, OFFAL, POLKA, ZONAL

💡 Next suggested word: CLOUT
📊 Information gain: 4.123 bits
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

The solver implements:

1. **Feedback Simulation**: For each possible guess, simulates all possible feedback outcomes
2. **Entropy Calculation**: Uses Shannon entropy to measure information content
3. **Solution Filtering**: Maintains a list of words compatible with all previous feedback
4. **Information Maximization**: Always chooses the guess that provides the most expected information

This approach typically solves Wordle puzzles in 3-4 guesses on average!

## Requirements

- Python 3.6+
- `wordles.txt` file with 5-letter words (one per line, uppercase)
- Terminal that supports ANSI color codes 

