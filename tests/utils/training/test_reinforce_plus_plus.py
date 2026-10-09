# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from relax.utils.training.ppo_utils import compute_approx_kl as _compiled_compute_approx_kl
from relax.utils.training.ppo_utils import compute_policy_loss as _compiled_compute_policy_loss
from relax.utils.training.ppo_utils import (
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)


compute_approx_kl = torch.compiler.disable(_compiled_compute_approx_kl)
compute_policy_loss = torch.compiler.disable(_compiled_compute_policy_loss)


@pytest.fixture(autouse=True)
def fake_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_context_parallel_world_size=lambda: 1)
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)


def _reference_returns(reward, token_kl, mask, kl_coef, gamma):
    shaped_reward = torch.zeros_like(token_kl)
    for token_index in range(token_kl.numel()):
        if mask[token_index] != 0:
            shaped_reward[token_index] = -kl_coef * token_kl[token_index]
    valid_indices = [index for index in range(mask.numel()) if mask[index] != 0]
    if not valid_indices:
        return torch.zeros_like(token_kl)
    shaped_reward[valid_indices[-1]] += reward

    expected = torch.zeros_like(token_kl)
    running = token_kl.new_zeros(())
    for token_index in range(token_kl.numel() - 1, -1, -1):
        running = shaped_reward[token_index] + gamma * running
        if mask[token_index] != 0:
            expected[token_index] = running
    return expected


def test_returns_match_independent_reference_for_variable_lengths_and_masks():
    rewards = torch.tensor([1.5, -0.25, 2.0], dtype=torch.float64)
    token_kls = [
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([0.1, 9999.0, -0.3], dtype=torch.float64),
        torch.tensor([0.2, -0.4, 0.7, -8888.0, 0.5], dtype=torch.float64),
    ]
    masks = [
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float64),
    ]
    kl_coef = 0.2
    gamma = 1.0

    actual = get_reinforce_plus_plus_returns(
        rewards,
        token_kls,
        masks,
        response_lengths=[1, 3, 5],
        total_lengths=[1, 3, 5],
        kl_coef=kl_coef,
        gamma=gamma,
    )
    expected = [
        _reference_returns(reward, token_kl, mask, kl_coef, gamma)
        for reward, token_kl, mask in zip(rewards, token_kls, masks, strict=True)
    ]

    for actual_tensor, expected_tensor, mask in zip(actual, expected, masks, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-6, rtol=1e-6)
        assert torch.equal(actual_tensor[mask == 0], torch.zeros_like(actual_tensor[mask == 0]))


def test_returns_support_discounting_without_leaking_masked_kl():
    reward = torch.tensor([1.0], dtype=torch.float64)
    token_kl = [torch.tensor([0.5, 12345.0, -0.25], dtype=torch.float64)]
    mask = [torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)]

    actual = get_reinforce_plus_plus_returns(reward, token_kl, mask, [3], [3], 0.1, 0.5)
    expected = [_reference_returns(reward[0], token_kl[0], mask[0], 0.1, 0.5)]

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)


def test_returns_make_fully_masked_local_response_zero():
    actual = get_reinforce_plus_plus_returns(
        torch.tensor([0.0]),
        [torch.tensor([float("nan"), float("inf")])],
        [torch.zeros(2)],
        [2],
        [2],
        0.1,
        1.0,
    )
    torch.testing.assert_close(actual[0], torch.zeros(2), atol=0, rtol=0)


def test_returns_ignore_nonfinite_values_outside_mask():
    reward = torch.tensor([1.0], dtype=torch.float64)
    token_kl = [torch.tensor([0.5, float("nan"), -0.25, float("inf")], dtype=torch.float64)]
    mask = [torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)]

    actual = get_reinforce_plus_plus_returns(reward, token_kl, mask, [4], [4], 0.1, 1.0)
    expected = [_reference_returns(reward[0], token_kl[0], mask[0], 0.1, 1.0)]

    torch.testing.assert_close(actual[0], expected[0], atol=1e-12, rtol=0)
    assert torch.isfinite(actual[0]).all()


def test_all_zero_reward_and_kl_produce_finite_zero_returns():
    actual = get_reinforce_plus_plus_returns(
        rewards=torch.zeros(2),
        kl=[torch.zeros(1), torch.zeros(3)],
        loss_masks=[torch.ones(1), torch.tensor([1.0, 1.0, 0.0])],
        response_lengths=[1, 3],
        total_lengths=[1, 3],
        kl_coef=0.1,
        gamma=1.0,
    )

    for returns in actual:
        torch.testing.assert_close(returns, torch.zeros_like(returns), atol=0, rtol=0)
        assert torch.isfinite(returns).all()


def _legacy_loop_returns(reward, token_kl, mask, kl_coef, gamma):
    """The per-token recurrence that the vectorized implementation replaced."""
    valid_mask = mask != 0
    if not torch.any(valid_mask):
        return torch.zeros_like(token_kl)
    token_level_rewards = -kl_coef * torch.where(valid_mask, token_kl, torch.zeros_like(token_kl))
    token_level_rewards[valid_mask.nonzero(as_tuple=True)[0][-1]] += reward
    returns = torch.zeros_like(token_level_rewards)
    running = 0.0
    for t in reversed(range(token_level_rewards.size(0))):
        running = token_level_rewards[t] + gamma * running
        returns[t] = running
    return torch.where(valid_mask, returns, torch.zeros_like(returns))


@pytest.mark.parametrize("gamma", [1.0, 0.99, 0.5, 0.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_returns_match_reference_on_random_batches(gamma, dtype):
    generator = torch.Generator().manual_seed(1234)
    lengths = [1, 2, 7, 128, 129, 300, 0, 5]
    rewards = torch.randn(len(lengths), generator=generator, dtype=dtype)
    token_kls, masks = [], []
    for index, length in enumerate(lengths):
        token_kls.append(torch.randn(length, generator=generator, dtype=dtype))
        mask = (torch.rand(length, generator=generator) > 0.2).to(dtype)
        if index == 3:
            mask[-17:] = 0  # masked tail: the reward must land on the last valid token
        if index == 7:
            mask.zero_()  # fully masked response
        masks.append(mask)

    actual = get_reinforce_plus_plus_returns(rewards, token_kls, masks, lengths, lengths, kl_coef=0.05, gamma=gamma)

    # Against the float64 reference the result is correctly rounded; the legacy
    # loop accumulated in the input dtype, so float32 differs by its rounding error.
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    legacy_tolerance = 1e-4 if dtype == torch.float32 else 1e-12
    for reward, token_kl, mask, actual_tensor in zip(rewards, token_kls, masks, actual, strict=True):
        assert actual_tensor.dtype == dtype
        assert actual_tensor.shape == token_kl.shape
        expected = _reference_returns(reward.double(), token_kl.double(), mask, 0.05, gamma).to(dtype)
        torch.testing.assert_close(actual_tensor, expected, atol=tolerance, rtol=tolerance)
        legacy = _legacy_loop_returns(reward, token_kl, mask, 0.05, gamma)
        torch.testing.assert_close(actual_tensor, legacy, atol=legacy_tolerance, rtol=legacy_tolerance)
        assert torch.equal(actual_tensor[mask == 0], torch.zeros_like(actual_tensor[mask == 0]))


def test_returns_handle_empty_response():
    actual = get_reinforce_plus_plus_returns(
        torch.tensor([3.0]), [torch.zeros(0)], [torch.zeros(0)], [0], [0], 0.1, 1.0
    )
    assert actual[0].shape == (0,)


def test_reverse_discounted_cumsum_edge_cases():
    from relax.utils.training.ppo_utils import _reverse_discounted_cumsum

    values = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(_reverse_discounted_cumsum(values, 1.0), torch.tensor([6.0, 5.0, 3.0]))
    torch.testing.assert_close(_reverse_discounted_cumsum(values, 0.5), torch.tensor([2.75, 3.5, 3.0]))
    torch.testing.assert_close(_reverse_discounted_cumsum(values, 0.0), values)
    counts = _reverse_discounted_cumsum(torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32), 1.0)
    assert counts.dtype == torch.int32
    assert counts.tolist() == [3, 2, 2, 1, 0]
    assert _reverse_discounted_cumsum(torch.zeros(0), 0.9).shape == (0,)
    with pytest.raises(ValueError, match="gamma == 1"):
        _reverse_discounted_cumsum(torch.ones(2, dtype=torch.int64), 0.9)


def test_returns_validate_response_batch_lengths_and_shapes():
    with pytest.raises(ValueError, match="same number of responses"):
        get_reinforce_plus_plus_returns(torch.tensor([0.0, 1.0]), [torch.ones(1)], [torch.ones(1)], [1], [1], 0.1, 1.0)
    with pytest.raises(ValueError, match="must have the same shape"):
        get_reinforce_plus_plus_returns(torch.tensor([0.0]), [torch.ones(2)], [torch.ones(1)], [1], [1], 0.1, 1.0)


def test_baseline_advantages_are_masked_and_do_not_depend_on_token_kl():
    centered_rewards = torch.tensor([1.0, -1.0], dtype=torch.float64)
    token_shapes = [
        torch.tensor([5.0, 9999.0, -7.0], dtype=torch.float64),
        torch.tensor([-123.0, 456.0], dtype=torch.float64),
    ]
    masks = [
        torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    ]

    actual = get_reinforce_plus_plus_baseline_advantages(centered_rewards, token_shapes, masks)
    expected = [
        torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([0.0, -1.0], dtype=torch.float64),
    ]

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


def test_baseline_advantages_ignore_nonfinite_shape_values_outside_mask():
    actual = get_reinforce_plus_plus_baseline_advantages(
        torch.tensor([2.0], dtype=torch.float64),
        [torch.tensor([0.0, float("nan"), float("inf")], dtype=torch.float64)],
        [torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)],
    )
    torch.testing.assert_close(actual[0], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64), atol=0, rtol=0)


def test_baseline_advantages_validate_batch_lengths():
    with pytest.raises(ValueError, match="same number of responses"):
        get_reinforce_plus_plus_baseline_advantages(torch.tensor([1.0, -1.0]), [torch.ones(2)], [torch.ones(2)])
    with pytest.raises(ValueError, match="must match"):
        get_reinforce_plus_plus_baseline_advantages(torch.tensor([1.0]), [torch.ones(2)], [torch.ones(1)])


def test_token_policy_and_k2_losses_match_independent_formulas():
    current_log_probs = torch.tensor([-0.1, -1.2, -0.3, -2.0], dtype=torch.float64)
    old_log_probs = torch.tensor([-0.2, -0.8, -0.5, -2.4], dtype=torch.float64)
    ref_log_probs = torch.tensor([-0.4, -1.0, -0.1, -2.5], dtype=torch.float64)
    advantages = torch.tensor([1.0, -2.0, 0.5, -0.25], dtype=torch.float64)

    ppo_kl = old_log_probs - current_log_probs
    actual_pg, actual_clipfrac = compute_policy_loss(ppo_kl, advantages, 0.2, 0.2)

    ratio = torch.exp(current_log_probs - old_log_probs)
    expected_pg = torch.maximum(
        -ratio * advantages,
        -torch.clamp(ratio, 0.8, 1.2) * advantages,
    )
    expected_clipfrac = (-torch.clamp(ratio, 0.8, 1.2) * advantages > -ratio * advantages).to(dtype=torch.float32)
    expected_k2 = 0.5 * (current_log_probs.float() - ref_log_probs.float()).square()
    actual_k2 = compute_approx_kl(current_log_probs, ref_log_probs, "k2")

    torch.testing.assert_close(actual_pg, expected_pg, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_clipfrac, expected_clipfrac, atol=0, rtol=0)
    torch.testing.assert_close(actual_k2, expected_k2, atol=1e-6, rtol=1e-6)
