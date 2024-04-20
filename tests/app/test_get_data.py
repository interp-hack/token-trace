from token_trace.app.get_data import get_data


def test_get_data(text: str):
    df = get_data(text, force_rerun=True)
    assert not df.empty
