import pytest


@pytest.mark.xfail(reason="not implemented")
def test_loaders_exist():
    from src.data.onchain import load_active_addresses, load_exchange_netflow  # noqa: F401
    pytest.fail("not implemented")
