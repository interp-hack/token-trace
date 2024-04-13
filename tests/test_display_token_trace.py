def test_display_token_trace():
    from token_trace.display_token_trace import display_token_trace

    res = display_token_trace(
        tokens=[
            "I",
            "am",
            "a",
        ],
        layer_vals=[
            [
                [
                    (123, 0.5),
                    (124, 0.6),
                    (125, 0.7),
                    (126, 0.8),
                ],
                [
                    (123, 0.5),
                    (124, 0.6),
                    (125, 0.7),
                    (126, 0.8),
                ],
            ],
            [
                [
                    (123, 0.5),
                    (124, 0.6),
                    (125, 0.7),
                    (126, 0.8),
                ],
                [
                    (123, 0.5),
                    (124, 0.6),
                    (125, 0.7),
                    (126, 0.8),
                ],
            ],
        ],
    )
    assert res.src is not None
