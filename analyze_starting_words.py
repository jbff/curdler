#!/usr/bin/env python3
"""
Analyze all starting words for Wordle to find optimal choices for both normal and hard mode
"""

import math
from collections import Counter
from wordle_solver import WordleSolver, Colors

def analyze_starting_word(solver, word):
    """Analyze the information gain of a starting word."""
    if word not in solver.words:
        return None
    
    information_gain = solver._calculate_information_gain(word, solver.words)
    
    # Analyze letter distribution
    letter_counts = Counter(word)
    vowels = sum(1 for c in word if c in 'AEIOU')
    consonants = 5 - vowels
    
    # Calculate letter frequency score (how common the letters are)
    letter_frequency = {
        'E': 12.0, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 'N': 6.7, 'S': 6.3, 'H': 6.1,
        'R': 6.0, 'D': 4.3, 'L': 4.0, 'C': 2.8, 'U': 2.8, 'M': 2.4, 'W': 2.4, 'F': 2.2,
        'G': 2.0, 'Y': 2.0, 'P': 1.9, 'B': 1.5, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
        'Q': 0.10, 'Z': 0.07
    }
    
    frequency_score = sum(letter_frequency.get(c, 0) for c in word)
    
    # Analyze feedback distribution
    feedback_counts = Counter()
    for solution in solver.words:
        feedback = solver._get_feedback(word, solution)
        feedback_str = ''.join(feedback)
        feedback_counts[feedback_str] += 1
    
    # Calculate how well the word splits the solution space
    max_group_size = max(feedback_counts.values()) if feedback_counts else 0
    split_efficiency = 1 - (max_group_size / len(solver.words))
    
    return {
        'word': word,
        'information_gain': information_gain,
        'vowels': vowels,
        'consonants': consonants,
        'frequency_score': frequency_score,
        'split_efficiency': split_efficiency,
        'feedback_distribution': feedback_counts
    }

def analyze_hard_mode_performance(solver, word, analysis_data):
    """Analyze how well a word performs in hard mode scenarios."""
    hard_mode_scores = []
    
    # Test various feedback scenarios
    test_scenarios = [
        ("XXXXX", "All gray"),
        ("XXYXX", "One yellow in middle"),
        ("GXXXX", "One green at start"),
        ("XXGXX", "One green in middle"),
        ("YXXXX", "One yellow at start"),
        ("GGXXX", "Two greens at start"),
        ("XXGGX", "Two greens in middle"),
        ("GXGXX", "Two greens separated"),
    ]
    
    for feedback_str, description in test_scenarios:
        feedback = list(feedback_str)
        
        # Create a temporary solver to test this scenario
        temp_solver = WordleSolver(hard_mode=True)
        temp_solver.guess_history.append((word, feedback))
        
        # Count how many valid hard mode guesses remain
        valid_guesses = [w for w in temp_solver.words if temp_solver._is_hard_mode_compatible(w)]
        
        # Score this scenario (fewer remaining guesses is better for hard mode)
        scenario_score = len(valid_guesses)
        hard_mode_scores.append(scenario_score)
    
    # Calculate average hard mode performance
    avg_hard_mode_score = sum(hard_mode_scores) / len(hard_mode_scores)
    analysis_data['hard_mode_score'] = avg_hard_mode_score
    analysis_data['hard_mode_scenarios'] = hard_mode_scores

def main():
    solver = WordleSolver()
    
    print(f"{Colors.BOLD}{Colors.PURPLE}🔍 Comprehensive Wordle Starting Word Analysis{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.WHITE}Total solutions: {Colors.BOLD}{len(solver.words)}{Colors.END}")
    print(f"{Colors.WHITE}Analyzing all words for optimal starting choices...{Colors.END}")
    print()
    
    # Analyze all words (this will take some time)
    print(f"{Colors.YELLOW}⏳ Analyzing all {len(solver.words)} words...{Colors.END}")
    
    all_results = []
    progress_interval = max(1, len(solver.words) // 20)  # Show progress every 5%
    
    for i, word in enumerate(solver.words):
        if i % progress_interval == 0:
            progress = (i / len(solver.words)) * 100
            print(f"{Colors.CYAN}Progress: {progress:.1f}% ({i}/{len(solver.words)}){Colors.END}")
        
        analysis = analyze_starting_word(solver, word)
        if analysis:
            analyze_hard_mode_performance(solver, word, analysis)
            all_results.append(analysis)
    
    print(f"{Colors.GREEN}✅ Analysis complete!{Colors.END}")
    print()
    
    # Sort by different criteria
    by_info_gain = sorted(all_results, key=lambda x: x['information_gain'], reverse=True)
    by_hard_mode = sorted(all_results, key=lambda x: x['hard_mode_score'])
    by_frequency = sorted(all_results, key=lambda x: x['frequency_score'], reverse=True)
    by_split_efficiency = sorted(all_results, key=lambda x: x['split_efficiency'], reverse=True)
    
    # Display top results for normal mode
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}📈 TOP 20 WORDS BY INFORMATION GAIN (Normal Mode){Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")
    
    for i, result in enumerate(by_info_gain[:20], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        word = result['word']
        info_gain = result['information_gain']
        vowels = result['vowels']
        consonants = result['consonants']
        split_eff = result['split_efficiency']
        
        print(f"{medal} {Colors.BOLD}{word}{Colors.END}: {Colors.BLUE}{info_gain:.3f}{Colors.END} bits")
        print(f"    {Colors.CYAN}Letters:{Colors.END} {', '.join(word)} | {Colors.CYAN}Vowels:{Colors.END} {vowels} | {Colors.CYAN}Split efficiency:{Colors.END} {split_eff:.3f}")
    
    # Display top results for hard mode
    print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}🔒 TOP 20 WORDS FOR HARD MODE{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.WHITE}Lower scores are better (fewer remaining guesses on average){Colors.END}")
    
    for i, result in enumerate(by_hard_mode[:20], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        word = result['word']
        hard_mode_score = result['hard_mode_score']
        info_gain = result['information_gain']
        
        print(f"{medal} {Colors.BOLD}{word}{Colors.END}: {Colors.YELLOW}{hard_mode_score:.1f}{Colors.END} avg remaining guesses")
        print(f"    {Colors.CYAN}Info gain:{Colors.END} {info_gain:.3f} bits | {Colors.CYAN}Letters:{Colors.END} {', '.join(word)}")
    
    # Display balanced recommendations
    print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}⚖️  BALANCED RECOMMENDATIONS{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")
    
    # Find words that are good in both modes
    top_info_words = set(result['word'] for result in by_info_gain[:50])
    top_hard_words = set(result['word'] for result in by_hard_mode[:50])
    balanced_words = top_info_words.intersection(top_hard_words)
    
    if balanced_words:
        print(f"{Colors.GREEN}Words that perform well in both modes:{Colors.END}")
        for word in sorted(balanced_words)[:10]:
            info_result = next(r for r in by_info_gain if r['word'] == word)
            hard_result = next(r for r in by_hard_mode if r['word'] == word)
            print(f"  {Colors.BOLD}{word}{Colors.END}: {Colors.BLUE}{info_result['information_gain']:.3f}{Colors.END} bits, {Colors.YELLOW}{hard_result['hard_mode_score']:.1f}{Colors.END} avg remaining")
    
    # Final recommendations
    print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}🏆 FINAL RECOMMENDATIONS{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")
    
    best_normal = by_info_gain[0]
    best_hard = by_hard_mode[0]
    
    print(f"{Colors.GREEN}For Normal Mode:{Colors.END}")
    print(f"  {Colors.BOLD}{best_normal['word']}{Colors.END} - {Colors.BLUE}{best_normal['information_gain']:.3f}{Colors.END} bits of information")
    print(f"  Letters: {', '.join(best_normal['word'])} | Vowels: {best_normal['vowels']}")
    
    print(f"\n{Colors.YELLOW}For Hard Mode:{Colors.END}")
    print(f"  {Colors.BOLD}{best_hard['word']}{Colors.END} - {Colors.YELLOW}{best_hard['hard_mode_score']:.1f}{Colors.END} avg remaining guesses")
    print(f"  Letters: {', '.join(best_hard['word'])} | Vowels: {best_hard['vowels']}")
    
    if best_normal['word'] == best_hard['word']:
        print(f"\n{Colors.GREEN}🎉 {best_normal['word']} is optimal for both modes!{Colors.END}")
    else:
        print(f"\n{Colors.CYAN}💡 Consider using different words for different modes{Colors.END}")

if __name__ == "__main__":
    main() 