import torch

from src.logprobs import compute_token_logprobs


def test_compute_token_logprobs_gathers_target_token_logprobs() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, 1.0],
            ]
        ]
    )
    target_ids = torch.tensor([[0, 2]])
    mask = torch.tensor([[1.0, 1.0]])

    result = compute_token_logprobs(logits, target_ids, mask)

    expected = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(result.token_logprobs, expected)


def test_compute_token_logprobs_sums_only_masked_tokens_per_sequence() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [-1.0, 0.0, 2.0],
                [4.0, 1.0, 0.0],
                [0.0, -2.0, 2.0],
            ],
        ]
    )
    target_ids = torch.tensor(
        [
            [0, 2, 1],
            [2, 0, 1],
        ]
    )
    mask = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    result = compute_token_logprobs(logits, target_ids, mask)

    expected_token_logprobs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    expected_sequence_logprobs = (expected_token_logprobs * mask).sum(dim=1)
    torch.testing.assert_close(result.token_logprobs, expected_token_logprobs)
    torch.testing.assert_close(result.sequence_logprobs, expected_sequence_logprobs)
    torch.testing.assert_close(result.mask, mask)


def test_compute_token_logprobs_returns_zero_sequence_logprob_when_mask_is_empty() -> None:
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target_ids = torch.tensor([[0, 1]])
    mask = torch.tensor([[0.0, 0.0]])

    result = compute_token_logprobs(logits, target_ids, mask)

    torch.testing.assert_close(result.sequence_logprobs, torch.tensor([0.0]))
