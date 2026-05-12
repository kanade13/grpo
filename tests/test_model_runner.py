from src.model_runner import ModelRunner


def test_trim_generated_token_ids_stops_at_first_stop_token() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.stop_token_ids = {0, 2}

    trimmed, stopped_early = runner._trim_generated_token_ids([10, 11, 2, 0, 0])

    assert trimmed == [10, 11, 2]
    assert stopped_early is True


def test_trim_generated_token_ids_keeps_full_output_without_stop_token() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.stop_token_ids = {0, 2}

    trimmed, stopped_early = runner._trim_generated_token_ids([10, 11, 12])

    assert trimmed == [10, 11, 12]
    assert stopped_early is False
