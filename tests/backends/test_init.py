import fast_carpenter.backends as backends


def test_get_backend():
    alphatwirl_back = backends.get_backend("alphatwirl")
    assert hasattr(alphatwirl_back, "execute")