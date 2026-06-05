from src.genai_eval.numeric_consistency import numeric_consistency


def test_numeric_consistency_accepts_commas_and_rounding():
    text = 'Revenue gap is 1,234.50 and confidence is 0.83.'
    assert numeric_consistency(text, [1234.50, 0.83]) == 1.0
