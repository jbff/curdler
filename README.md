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

## 🔍 Comprehensive Starting Word Analysis

Based on information theory analysis of all 2,331 possible Wordle solutions:

### 📈 TOP 20 WORDS BY INFORMATION GAIN (Normal Mode)

🥇 **RAISE**: 5.881 bits  
    Letters: R, A, I, S, E | Vowels: 3 | Split efficiency: 0.928

🥈 **SLATE**: 5.861 bits  
    Letters: S, L, A, T, E | Vowels: 2 | Split efficiency: 0.904

🥉 **IRATE**: 5.834 bits  
    Letters: I, R, A, T, E | Vowels: 3 | Split efficiency: 0.917

4. **CRATE**: 5.830 bits  
    Letters: C, R, A, T, E | Vowels: 2 | Split efficiency: 0.893

5. **TRACE**: 5.827 bits  
    Letters: T, R, A, C, E | Vowels: 2 | Split efficiency: 0.893

6. **ARISE**: 5.824 bits  
    Letters: A, R, I, S, E | Vowels: 3 | Split efficiency: 0.928

7. **STARE**: 5.816 bits  
    Letters: S, T, A, R, E | Vowels: 2 | Split efficiency: 0.903

8. **SNARE**: 5.772 bits  
    Letters: S, N, A, R, E | Vowels: 2 | Split efficiency: 0.905

9. **AROSE**: 5.771 bits  
    Letters: A, R, O, S, E | Vowels: 3 | Split efficiency: 0.921

10. **LEAST**: 5.757 bits  
    Letters: L, E, A, S, T | Vowels: 2 | Split efficiency: 0.904

11. **STALE**: 5.747 bits  
    Letters: S, T, A, L, E | Vowels: 2 | Split efficiency: 0.904

12. **ALERT**: 5.743 bits  
    Letters: A, L, E, R, T | Vowels: 2 | Split efficiency: 0.915

13. **SANER**: 5.735 bits  
    Letters: S, A, N, E, R | Vowels: 2 | Split efficiency: 0.905

14. **CRANE**: 5.734 bits  
    Letters: C, R, A, N, E | Vowels: 2 | Split efficiency: 0.885

15. **ALTER**: 5.711 bits  
    Letters: A, L, T, E, R | Vowels: 2 | Split efficiency: 0.915

16. **LATER**: 5.706 bits  
    Letters: L, A, T, E, R | Vowels: 2 | Split efficiency: 0.915

17. **REACT**: 5.691 bits  
    Letters: R, E, A, C, T | Vowels: 2 | Split efficiency: 0.893

18. **TRADE**: 5.684 bits  
    Letters: T, R, A, D, E | Vowels: 2 | Split efficiency: 0.884

19. **LEANT**: 5.683 bits  
    Letters: L, E, A, N, T | Vowels: 2 | Split efficiency: 0.910

20. **ROAST**: 5.652 bits  
    Letters: R, O, A, S, T | Vowels: 2 | Split efficiency: 0.886

### 🔒 TOP 20 WORDS FOR HARD MODE

*Lower scores are better (fewer remaining guesses on average)*

🥇 **CANOE**: 47.8 avg remaining guesses  
    Info gain: 5.509 bits | Letters: C, A, N, O, E

🥈 **CAMEO**: 49.2 avg remaining guesses  
    Info gain: 5.065 bits | Letters: C, A, M, E, O

🥉 **SEPIA**: 50.6 avg remaining guesses  
    Info gain: 5.166 bits | Letters: S, E, P, I, A

4. **MEDIA**: 52.5 avg remaining guesses  
    Info gain: 4.785 bits | Letters: M, E, D, I, A

5. **RATIO**: 53.2 avg remaining guesses  
    Info gain: 5.356 bits | Letters: R, A, T, I, O

6. **LAYER**: 55.5 avg remaining guesses  
    Info gain: 5.359 bits | Letters: L, A, Y, E, R

7. **HATER**: 56.2 avg remaining guesses  
    Info gain: 5.480 bits | Letters: H, A, T, E, R

8. **HAUTE**: 56.6 avg remaining guesses  
    Info gain: 5.283 bits | Letters: H, A, U, T, E

9. **RADIO**: 56.6 avg remaining guesses  
    Info gain: 5.059 bits | Letters: R, A, D, I, O

10. **VALUE**: 56.8 avg remaining guesses  
    Info gain: 4.886 bits | Letters: V, A, L, U, E

11. **LASER**: 57.4 avg remaining guesses  
    Info gain: 5.629 bits | Letters: L, A, S, E, R

12. **RAISE**: 57.4 avg remaining guesses  
    Info gain: 5.881 bits | Letters: R, A, I, S, E

13. **VAGUE**: 57.5 avg remaining guesses  
    Info gain: 4.504 bits | Letters: V, A, G, U, E

14. **PATIO**: 57.8 avg remaining guesses  
    Info gain: 4.974 bits | Letters: P, A, T, I, O

15. **SAUTE**: 58.0 avg remaining guesses  
    Info gain: 5.637 bits | Letters: S, A, U, T, E

16. **PAYER**: 58.1 avg remaining guesses  
    Info gain: 5.166 bits | Letters: P, A, Y, E, R

17. **GAYER**: 58.8 avg remaining guesses  
    Info gain: 5.079 bits | Letters: G, A, Y, E, R

18. **LATER**: 58.8 avg remaining guesses  
    Info gain: 5.706 bits | Letters: L, A, T, E, R

19. **PASTE**: 58.8 avg remaining guesses  
    Info gain: 5.429 bits | Letters: P, A, S, T, E

20. **DELTA**: 59.1 avg remaining guesses  
    Info gain: 5.238 bits | Letters: D, E, L, T, A

### ⚖️ BALANCED RECOMMENDATIONS

*Words that perform well in both modes:*

- **ALTER**: 5.711 bits, 59.6 avg remaining
- **CASTE**: 5.570 bits, 60.4 avg remaining  
- **CATER**: 5.641 bits, 60.1 avg remaining
- **LASER**: 5.629 bits, 57.4 avg remaining
- **LATER**: 5.706 bits, 58.8 avg remaining
- **PARSE**: 5.636 bits, 61.1 avg remaining
- **RAISE**: 5.881 bits, 57.4 avg remaining
- **RENAL**: 5.549 bits, 61.5 avg remaining
- **SANER**: 5.735 bits, 59.1 avg remaining
- **SAUTE**: 5.637 bits, 58.0 avg remaining

### 🏆 FINAL RECOMMENDATIONS

**For Normal Mode:**  
🥇 **RAISE** - 5.881 bits of information  
Letters: R, A, I, S, E | Vowels: 3

**For Hard Mode:**  
🥇 **CANOE** - 47.8 avg remaining guesses  
Letters: C, A, N, O, E | Vowels: 3

💡 *Consider using different words for different modes*

