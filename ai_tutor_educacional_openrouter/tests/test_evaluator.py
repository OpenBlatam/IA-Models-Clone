"""
Tests for AnswerEvaluator.
"""

import pytest
from core.evaluator import AnswerEvaluator, EvaluationResult


@pytest.fixture
def evaluator():
    """Create an evaluator instance."""
    return AnswerEvaluator()


def test_evaluate_multiple_choice_correct(evaluator):
    """Test evaluating correct multiple choice answer."""
    result = evaluator.evaluate_answer(
        student_answer="A",
        correct_answer="A",
        question_type="multiple_choice"
    )
    
    assert result.score == 1.0
    assert result.percentage == 100.0
    assert "Correcto" in result.feedback


def test_evaluate_multiple_choice_incorrect(evaluator):
    """Test evaluating incorrect multiple choice answer."""
    result = evaluator.evaluate_answer(
        student_answer="B",
        correct_answer="A",
        question_type="multiple_choice"
    )
    
    assert result.score == 0.0
    assert result.percentage == 0.0
    assert "Incorrecto" in result.feedback


def test_evaluate_short_answer_similar(evaluator):
    """Test evaluating similar short answer."""
    result = evaluator.evaluate_answer(
        student_answer="The process by which plants make food",
        correct_answer="photosynthesis",
        question_type="short_answer"
    )
    
    assert result.score > 0.0
    assert result.score <= 1.0


def test_evaluate_quiz(evaluator):
    """Test evaluating a complete quiz."""
    student_answers = {
        "q1": "A",
        "q2": "B",
        "q3": "photosynthesis"
    }
    
    quiz_answers = {
        "q1": {"correct_answer": "A", "type": "multiple_choice"},
        "q2": {"correct_answer": "A", "type": "multiple_choice"},
        "q3": {"correct_answer": "photosynthesis", "type": "short_answer"}
    }
    
    result = evaluator.evaluate_quiz(student_answers, quiz_answers)
    
    assert "total_score" in result
    assert "max_score" in result
    assert "percentage" in result
    assert "grade" in result
    assert result["max_score"] == 3






