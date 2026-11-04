from scripts.train_goemotions_model import limit_split


def test_limit_split_uses_reproducible_random_subset_and_keeps_labels_aligned():
    texts = [f"example {index}" for index in range(100)]
    targets = [[float(index)] for index in range(100)]

    first = limit_split(texts, targets, limit=20, seed=42)
    second = limit_split(texts, targets, limit=20, seed=42)

    assert first == second
    assert len(first[0]) == 20
    assert first[0] != texts[:20]
    assert first[1] == [[float(int(text.split()[-1]))] for text in first[0]]


def test_limit_split_keeps_full_data_when_no_limit_or_limit_exceeds_size():
    texts = ["one", "two"]
    targets = [[1.0], [0.0]]

    assert limit_split(texts, targets, limit=None, seed=42) == (texts, targets)
    assert limit_split(texts, targets, limit=5, seed=42) == (texts, targets)


def test_limit_split_rejects_mismatched_rows():
    try:
        limit_split(["one"], [], limit=1, seed=42)
    except ValueError as error:
        assert "equal lengths" in str(error)
    else:
        raise AssertionError("Mismatched texts and targets must fail validation.")
