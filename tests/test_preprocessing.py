import pytest
from preprocessing import encode_extra_activities, preprocess_features, FEATURES


def test_encode_valid():
    assert encode_extra_activities("Yes") == 1

def test_encode_valid():
    assert encode_extra_activities("No") == 0


@pytest.mark.parametrize("value", ["Maybe", "yes", "", None])
def test_encode_invalid(value):
    with pytest.raises(ValueError):
        encode_extra_activities(value)


def test_preprocess_correct_column_order():
    # Give dictionary in WRONG order intentionally
    features = {
        "Previous Scores": 75.0,
        "Sleep Hours": 7.0,
        "Hours Studied": 6.0,
        "Sample Question Papers Practiced": 4.0,
        "Extracurricular Activities": "Yes",
    }


    X = preprocess_features(features)
    assert list(X.columns) == FEATURES


def test_preprocess_missing_column():
    #missing "Sample Question Papers Practiced"
    features_missing = {
        "Hours Studied": 6.0,
        "Previous Scores": 75.0,
        "Extracurricular Activities": "Yes",
        "Sleep Hours": 7.0,
    }

    with pytest.raises(ValueError) as err:
        preprocess_features(features_missing)

    assert "Missing columns" in str(err.value)