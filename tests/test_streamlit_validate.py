import pytest
from streamlit_validate import validate_input


def test_valid_inputs():
    assert validate_input(6.0, 75.0, "Yes", 7.0, 4.0) == []
    

@pytest.mark.parametrize("hrs_studied", [-1.0, 25.0])
def test_invalid_hours(hrs_studied):
    errors = validate_input(hrs_studied, 75.0, "Yes", 7.0, 4.0)
    assert "Hrs Studied must be between 0 and 24" in errors


@pytest.mark.parametrize("prev_score", [-10.0, 120.0])
def test_invalid_previous_scores(prev_score):
    errors = validate_input(6.0, prev_score, "No", 7.0, 4.0)
    assert "Scores must be between 0 and 100" in errors


def test_invalid_extra_value():
    errors = validate_input(6.0, 75.0, "Maybe", 7.0, 4.0)
    assert "Extracurricular activities must be Yes or No" in errors


@pytest.mark.parametrize("sleep_hrs", [-2.0, 30.0])
def test_invalid_sleep_hours(sleep_hrs):
    errors = validate_input(6.0, 75.0, "Yes", sleep_hrs, 4.0)
    assert "Sleep hours must be between 0 and 24" in errors


@pytest.mark.parametrize("sample_questions", [-2.0])
def test_invalid_papers(sample_questions):
    errors = validate_input(6.0, 75.0, "Yes", 7.0, sample_questions)
    assert "Sample Questions must be a positive number" in errors
