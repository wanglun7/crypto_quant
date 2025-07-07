import pytest


@pytest.mark.xfail(reason="not implemented")
def test_order_book_feature_exists():
    from src.features.micro import order_book_imbalance  # noqa: F401
    pytest.fail("not implemented")
