# Resource: https://docs.pytest.org/en/6.2.x/getting-started.html


def func(x):
    return x + 1


def test_answer():
    assert func(4) == 5


def main():
    print("Running")


if __name__ == "__main__":
    main()
    test_answer()  # pytest pytest_1 on terminal
