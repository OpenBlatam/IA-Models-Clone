"""
Benchmark Utilities - Comprehensive shared utilities for benchmarks.

Provides:
- Prompt formatting (multiple choice, few-shot)
- Answer extraction (letters, numbers, text)
- Answer evaluation (multiple choice, numeric, text matching)
- Few-shot example management
- Similarity calculations
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from difflib import SequenceMatcher
from collections import Counter

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# PROMPT FORMATTING
# ════════════════════════════════════════════════════════════════════════════════

def format_multiple_choice_options(
    choices: List[str],
    start_char: str = "A",
    numbered: bool = True
) -> str:
    """
    Format choices as multiple choice options.
    
    Args:
        choices: List of choice strings
        start_char: Starting character (default: "A")
        numbered: Whether to number options (default: True)
    
    Returns:
        Formatted options string
    
    Example:
        >>> format_multiple_choice_options(["Yes", "No"])
        'A. Yes\nB. No'
    """
    if not choices:
        return ""
    
    if numbered:
        return "\n".join([
            f"{chr(ord(start_char) + i)}. {choice}"
            for i, choice in enumerate(choices)
        ])
    else:
        return "\n".join([
            f"{chr(ord(start_char) + i)}. {choice}"
            for i, choice in enumerate(choices)
        ])


def format_question_with_options(
    question: str,
    choices: List[str],
    instruction: Optional[str] = None
) -> str:
    """
    Format a question with multiple choice options.
    
    Args:
        question: Question text
        choices: List of choice strings
        instruction: Optional instruction text
    
    Returns:
        Formatted prompt
    """
    parts = []
    
    if instruction:
        parts.append(instruction)
    
    parts.append(question)
    parts.append("")
    parts.append(format_multiple_choice_options(choices))
    parts.append("Answer:")
    
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════════
# ANSWER EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_letter_answer(
    prediction: str,
    valid_letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
) -> Optional[str]:
    """
    Extract letter answer (A, B, C, D, etc.) from prediction.
    
    Uses multiple strategies:
    1. Find letter at word boundary
    2. Check if starts with valid letter
    3. Find letter in parentheses
    4. Find letter after "Answer:" or similar markers
    
    Args:
        prediction: Model prediction
        valid_letters: Valid letter options
    
    Returns:
        Extracted letter or None
    
    Example:
        >>> extract_letter_answer("The answer is A")
        'A'
        >>> extract_letter_answer("Answer: (B)")
        'B'
    """
    if not prediction:
        return None
    
    prediction_upper = prediction.strip().upper()
    
    # Strategy 1: Find letter at word boundary (most common)
    pattern = rf'\b([{valid_letters}])\b'
    matches = re.findall(pattern, prediction_upper)
    if matches:
        # Return the first valid letter found
        return matches[0]
    
    # Strategy 2: Find letter in parentheses
    pattern = rf'\(([{valid_letters}])\)'
    match = re.search(pattern, prediction_upper)
    if match:
        return match.group(1)
    
    # Strategy 3: Find letter after "Answer:" or similar markers
    pattern = rf'(?:answer|choice|option|select)[:\s]+([{valid_letters}])'
    match = re.search(pattern, prediction_upper)
    if match:
        return match.group(1)
    
    # Strategy 4: Check if starts with valid letter
    if prediction_upper and prediction_upper[0] in valid_letters:
        return prediction_upper[0]
    
    return None


def extract_numeric_answer(
    prediction: str,
    tolerance: float = 0.01,
    prefer_last: bool = True
) -> Optional[float]:
    """
    Extract numeric answer from prediction.
    
    Supports multiple formats:
    - GSM8K format: "#### 42"
    - Plain numbers: "42" or "42.5"
    - In text: "The answer is 42"
    
    Args:
        prediction: Model prediction
        tolerance: Tolerance for floating point comparison (unused in extraction)
        prefer_last: If multiple numbers found, prefer the last one
    
    Returns:
        Extracted number or None
    
    Example:
        >>> extract_numeric_answer("The answer is #### 42")
        42.0
        >>> extract_numeric_answer("Result: 3.14")
        3.14
    """
    if not prediction:
        return None
    
    # Strategy 1: Try to find number after #### marker (GSM8K format)
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', prediction)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Strategy 2: Find number after "Answer:" or similar markers
    pattern = r'(?:answer|result|solution|value)[:\s]+([-+]?\d*\.?\d+)'
    match = re.search(pattern, prediction, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Strategy 3: Extract all numbers
    numbers = re.findall(r'[-+]?\d*\.?\d+', prediction)
    if numbers:
        try:
            if prefer_last:
                return float(numbers[-1])
            else:
                return float(numbers[0])
        except ValueError:
            pass
    
    return None


def extract_text_answer(
    prediction: str,
    max_length: int = 200
) -> str:
    """
    Extract text answer from prediction, cleaning it up.
    
    Args:
        prediction: Model prediction
        max_length: Maximum length of extracted answer
    
    Returns:
        Cleaned text answer
    """
    if not prediction:
        return ""
    
    # Remove common prefixes
    prefixes = [
        r'^(?:answer|result|solution|response)[:\s]+',
        r'^(?:the\s+)?answer\s+is\s+',
    ]
    
    text = prediction.strip()
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + "..."
    
    return text.strip()


# ════════════════════════════════════════════════════════════════════════════════
# TEXT SIMILARITY
# ════════════════════════════════════════════════════════════════════════════════

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using SequenceMatcher.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def calculate_word_overlap(text1: str, text2: str) -> float:
    """
    Calculate word overlap ratio between two texts.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0


def match_text_answer(
    prediction: str,
    correct_text: str,
    threshold: float = 0.5,
    method: str = "overlap"
) -> bool:
    """
    Match text answer using various similarity methods.
    
    Args:
        prediction: Model prediction
        correct_text: Correct answer text
        threshold: Minimum similarity threshold
        method: Similarity method ("exact", "overlap", "similarity")
    
    Returns:
        True if match found
    
    Example:
        >>> match_text_answer("The capital is Paris", "Paris", method="exact")
        True
    """
    prediction_lower = prediction.lower()
    correct_lower = correct_text.lower()
    
    # Exact match
    if method == "exact" or correct_lower in prediction_lower:
        return correct_lower in prediction_lower
    
    # Word overlap
    if method == "overlap":
        overlap = calculate_word_overlap(prediction, correct_text)
        return overlap >= threshold
    
    # Sequence similarity
    if method == "similarity":
        similarity = calculate_text_similarity(prediction, correct_text)
        return similarity >= threshold
    
    # Default: try all methods
    if correct_lower in prediction_lower:
        return True
    
    overlap = calculate_word_overlap(prediction, correct_text)
    if overlap >= threshold:
        return True
    
    similarity = calculate_text_similarity(prediction, correct_text)
    return similarity >= threshold


# ════════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES
# ════════════════════════════════════════════════════════════════════════════════

def format_few_shot_examples(
    examples: List[Dict[str, Any]],
    format_fn: Callable[[Dict[str, Any]], str],
    separator: str = "\n\n"
) -> str:
    """
    Format few-shot examples using a formatting function.
    
    Args:
        examples: List of example dictionaries
        format_fn: Function to format each example
        separator: Separator between examples
    
    Returns:
        Formatted few-shot examples string
    
    Example:
        >>> examples = [{"question": "Q1", "answer": "A1"}]
        >>> format_few_shot_examples(examples, lambda x: f"Q: {x['question']}\\nA: {x['answer']}")
        'Q: Q1\\nA: A1'
    """
    if not examples:
        return ""
    
    formatted = []
    for example in examples:
        try:
            formatted.append(format_fn(example))
        except (KeyError, TypeError) as e:
            logger.warning(f"Error formatting example: {e}")
            continue
    
    return separator.join(formatted)


def create_few_shot_prompt(
    instruction: str,
    few_shot_examples: List[Dict[str, Any]],
    current_example: Dict[str, Any],
    format_example_fn: Callable[[Dict[str, Any]], str],
    format_current_fn: Callable[[Dict[str, Any]], str],
    separator: str = "\n\n"
) -> str:
    """
    Create prompt with few-shot examples.
    
    Args:
        instruction: Instruction text
        few_shot_examples: List of few-shot examples
        current_example: Current example to evaluate
        format_example_fn: Function to format few-shot examples (with answer)
        format_current_fn: Function to format current example (without answer)
        separator: Separator between sections
    
    Returns:
        Complete prompt with few-shot examples
    
    Example:
        >>> instruction = "Answer the following questions."
        >>> few_shot = [{"q": "1+1?", "a": "2"}]
        >>> current = {"q": "2+2?"}
        >>> create_few_shot_prompt(
        ...     instruction, few_shot, current,
        ...     lambda x: f"Q: {x['q']}\\nA: {x['a']}",
        ...     lambda x: f"Q: {x['q']}\\nA:"
        ... )
    """
    parts = [instruction]
    
    if few_shot_examples:
        few_shot_text = format_few_shot_examples(
            few_shot_examples,
            format_example_fn,
            separator
        )
        if few_shot_text:
            parts.append(few_shot_text)
    
    parts.append(format_current_fn(current_example))
    
    return separator.join(parts)


def sample_few_shot_examples(
    dataset: List[Dict[str, Any]],
    num_examples: int,
    exclude: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample few-shot examples from dataset.
    
    Args:
        dataset: List of examples
        num_examples: Number of examples to sample
        exclude: Example to exclude from sampling
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled examples
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    available = [ex for ex in dataset if ex != exclude]
    
    if len(available) < num_examples:
        return available
    
    return random.sample(available, num_examples)


# ════════════════════════════════════════════════════════════════════════════════
# ANSWER EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_multiple_choice(
    prediction: str,
    correct_answer: Union[str, int],
    choices: Optional[List[str]] = None,
    strict: bool = False
) -> bool:
    """
    Evaluate multiple choice answer.
    
    Supports:
    - Letter answers (A, B, C, D)
    - Index answers (0, 1, 2, 3)
    - Text matching if choices provided
    
    Args:
        prediction: Model prediction
        correct_answer: Correct answer (letter, index, or text)
        choices: Optional list of choices for text matching
        strict: If True, only accept exact matches
    
    Returns:
        True if answer is correct
    
    Example:
        >>> evaluate_multiple_choice("The answer is A", "A")
        True
        >>> evaluate_multiple_choice("B", 1, ["Yes", "No"])
        True
    """
    # Normalize correct answer
    if isinstance(correct_answer, int):
        # Convert index to letter
        if 0 <= correct_answer < 26:
            correct_letter = chr(ord('A') + correct_answer)
        else:
            return False
    elif isinstance(correct_answer, str):
        correct_letter = correct_answer.upper()
    else:
        return False
    
    # Try letter matching first
    predicted_letter = extract_letter_answer(prediction)
    if predicted_letter:
        if strict:
            return predicted_letter == correct_letter
        else:
            # Allow case-insensitive comparison
            return predicted_letter.upper() == correct_letter.upper()
    
    # Try text matching if choices provided
    if choices:
        try:
            # Get correct choice text
            if len(correct_letter) == 1:
                correct_index = ord(correct_letter) - ord('A')
                if 0 <= correct_index < len(choices):
                    correct_text = choices[correct_index]
                    return match_text_answer(prediction, correct_text, threshold=0.5)
        except (ValueError, IndexError) as e:
            logger.debug(f"Error in text matching: {e}")
    
    return False


def evaluate_numeric_answer(
    prediction: str,
    correct_answer: Union[str, float, int],
    tolerance: float = 0.01,
    relative_tolerance: bool = False
) -> bool:
    """
    Evaluate numeric answer with tolerance.
    
    Args:
        prediction: Model prediction
        correct_answer: Correct answer (string or numeric)
        tolerance: Absolute tolerance for comparison
        relative_tolerance: If True, use relative tolerance (percentage)
    
    Returns:
        True if answer is correct
    
    Example:
        >>> evaluate_numeric_answer("42", 42.0)
        True
        >>> evaluate_numeric_answer("3.14", "3.14159", tolerance=0.01)
        True
    """
    # Extract numeric value from correct answer
    if isinstance(correct_answer, str):
        correct_match = re.search(r'[-+]?\d*\.?\d+', correct_answer)
        if not correct_match:
            return False
        try:
            correct_value = float(correct_match.group(0))
        except ValueError:
            return False
    else:
        correct_value = float(correct_answer)
    
    # Extract numeric value from prediction
    pred_value = extract_numeric_answer(prediction)
    if pred_value is None:
        return False
    
    # Compare with tolerance
    if relative_tolerance:
        if abs(correct_value) < 1e-10:
            return abs(pred_value) < tolerance
        else:
            relative_error = abs(pred_value - correct_value) / abs(correct_value)
            return relative_error < tolerance
    else:
        return abs(pred_value - correct_value) < tolerance


def evaluate_text_answer(
    prediction: str,
    correct_answers: List[str],
    incorrect_answers: Optional[List[str]] = None,
    threshold: float = 0.5
) -> bool:
    """
    Evaluate text answer against multiple correct/incorrect options.
    
    Args:
        prediction: Model prediction
        correct_answers: List of correct answer texts
        incorrect_answers: Optional list of incorrect answer texts
        threshold: Similarity threshold for matching
    
    Returns:
        True if answer matches a correct answer and doesn't match incorrect ones
    """
    prediction_lower = prediction.lower()
    
    # Check against incorrect answers first (higher priority)
    if incorrect_answers:
        for incorrect in incorrect_answers:
            if match_text_answer(prediction, incorrect, threshold=threshold):
                return False
    
    # Check against correct answers
    for correct in correct_answers:
        if match_text_answer(prediction, correct, threshold=threshold):
            return True
    
    return False


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Prompt formatting
    "format_multiple_choice_options",
    "format_question_with_options",
    # Answer extraction
    "extract_letter_answer",
    "extract_numeric_answer",
    "extract_text_answer",
    # Text similarity
    "calculate_text_similarity",
    "calculate_word_overlap",
    "match_text_answer",
    # Few-shot examples
    "format_few_shot_examples",
    "create_few_shot_prompt",
    "sample_few_shot_examples",
    # Answer evaluation
    "evaluate_multiple_choice",
    "evaluate_numeric_answer",
    "evaluate_text_answer",
]
