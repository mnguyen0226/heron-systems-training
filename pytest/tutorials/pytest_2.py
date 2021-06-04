# this script is for testing pytest in multiple test
def add_six(x):
    return x + 6


def test_1():
    assert add_six(1) == 7


# pytest pytest_1.py pytest_2.py
