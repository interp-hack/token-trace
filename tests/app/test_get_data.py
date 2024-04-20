from token_trace.app.get_circuit import get_circuit


def test_get_data(text: str):
    circuit = get_circuit(text, force_rerun=True)
    assert not circuit.node_ie_df.empty
