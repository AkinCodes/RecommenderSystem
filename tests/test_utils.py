import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import parse_duration


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("2 Seasons", 20),
        ("1 Season", 10),
        ("90 min", 90),
        ("1h 30m", 90),
        ("2h", 120),
        ("45m", 45),
        ("Duration: 3 seasons", 30),
        ("", 0),
        (None, 0),
        ("invalid text", 0),
    ],
)
def test_parse_duration_advanced(input_str, expected):
    assert parse_duration(input_str) == expected
