"""Pequeno pacote de exemplo para o teste de fumaça da pipeline."""


def classify_age(age):
    """Classify a person by age.

    Returns "child" for ages 0-12, "teen" for 13-17, "adult" for 18-64 and
    "senior" for 65 or older. Raises ValueError for negative ages.
    """
    if age < 0:
        raise ValueError("age must be non-negative")
    if age < 13:
        return "child"
    if age < 18:
        return "teen"
    if age < 65:
        return "adult"
    return "senior"


def can_register(age):
    """Return True if a user may register: only adults (age >= 18) can register.

    Raises ValueError for negative ages.
    """
    if age < 0:
        raise ValueError("age must be non-negative")
    return age >= 18
