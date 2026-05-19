import torch

from src.masks import build_response_mask


def test_build_response_mask_marks_only_response_tokens_before_padding() -> None:
    input_ids = torch.tensor(
        [
            [11, 12, 21, 22, 0, 0],
            [31, 41, 42, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    prompt_lengths = torch.tensor([2, 1], dtype=torch.long)
    response_lengths = torch.tensor([2, 2], dtype=torch.long)

    mask = build_response_mask(input_ids, prompt_lengths, response_lengths, pad_token_id=0)

    expected = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(mask, expected)


def test_build_response_mask_handles_batch_rows_with_different_prompt_lengths() -> None:
    input_ids = torch.tensor(
        [
            [10, 20, 21, 22],
            [30, 31, 40, 0],
            [50, 60, 0, 0],
        ],
        dtype=torch.long,
    )
    prompt_lengths = torch.tensor([1, 2, 2], dtype=torch.long)
    response_lengths = torch.tensor([3, 1, 0], dtype=torch.long)

    mask = build_response_mask(input_ids, prompt_lengths, response_lengths, pad_token_id=0)

    expected = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(mask, expected)


def test_build_response_mask_handles_empty_prompt() -> None:
    input_ids = torch.tensor([[21, 22, 0]], dtype=torch.long)
    prompt_lengths = torch.tensor([0], dtype=torch.long)
    response_lengths = torch.tensor([2], dtype=torch.long)

    mask = build_response_mask(input_ids, prompt_lengths, response_lengths, pad_token_id=0)

    expected = torch.tensor([[1, 1, 0]], dtype=torch.long)
    torch.testing.assert_close(mask, expected)


def test_build_response_mask_preserves_shape() -> None:
    input_ids = torch.tensor([[1, 2, 3], [4, 0, 0]], dtype=torch.long)
    prompt_lengths = torch.tensor([2, 1], dtype=torch.long)
    response_lengths = torch.tensor([1, 1], dtype=torch.long)

    mask = build_response_mask(input_ids, prompt_lengths, response_lengths, pad_token_id=0)

    assert mask.shape == input_ids.shape


def test_build_response_mask_keeps_real_eos_even_when_it_matches_pad_token() -> None:
    input_ids = torch.tensor([[11, 12, 21, 0, 0]], dtype=torch.long)
    prompt_lengths = torch.tensor([2], dtype=torch.long)
    response_lengths = torch.tensor([2], dtype=torch.long)

    mask = build_response_mask(input_ids, prompt_lengths, response_lengths, pad_token_id=0)

    expected = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.long)
    torch.testing.assert_close(mask, expected)
