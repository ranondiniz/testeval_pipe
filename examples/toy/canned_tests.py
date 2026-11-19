from toy_pkg.validators import classify_age


def test_child():
    assert classify_age(5) == "child"


def test_child_duplicate():
    # redundante com test_child
    assert classify_age(5) == "child"


def test_teen_lower_boundary():
    assert classify_age(13) == "teen"


def test_adult_lower_boundary():
    assert classify_age(18) == "adult"


def test_senior():
    assert classify_age(70) == "senior"


def test_negative_raises():
    try:
        classify_age(-1)
    except ValueError:
        return
    assert False, "esperava ValueError"


def test_wrong_oracle_teen():
    # oráculo ERRADO: 17 anos é "teen", não "adult"
    assert classify_age(17) == "adult"


def test_trivial():
    assert True
