import torch

from src.optim import build_adam_optimizer


def test_build_adam_optimizer_updates_parameter() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = build_adam_optimizer([parameter], lr=0.1)

    loss = parameter.square().sum()
    loss.backward()
    optimizer.step()

    assert parameter.item() < 1.0
