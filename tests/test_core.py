from laser.cholera import compute


def test_compute():
    assert compute(["a", "bc", "abc"]) == "abc"
