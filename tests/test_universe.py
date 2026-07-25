from stock_ai.data.universe import load_universe


def test_load_universe():
    tickers = load_universe()

    assert tickers
    assert tickers[0].code.endswith(".T")
