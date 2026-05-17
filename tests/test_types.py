from dataclasses import asdict

from src.types import GSM8KExample, GeneratedCompletion, PromptRollout, RewardResult


def test_dataclasses_can_be_constructed() -> None:
    example = GSM8KExample(
        id="test-0",
        question="What is 2+3?",
        answer="2+3=5\n#### 5",
        final_answer="5",
        prompt="What is 2+3?\nPlease reason step by step.",
    )
    reward = RewardResult(
        total_reward=1.1,
        answer_reward=1.0,
        format_reward=0.1,
        length_penalty=0.0,
        extracted_answer="5",
        is_correct=True,
    )
    completion = GeneratedCompletion(
        prompt_id=example.id,
        completion_id=0,
        prompt=example.prompt,
        response="\\boxed{5}",
        response_token_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        reward=reward,
    )
    rollout = PromptRollout(
        prompt_id=example.id,
        prompt=example.prompt,
        answer=example.final_answer,
        completions=[completion],
    )

    assert asdict(rollout)["completions"][0]["reward"]["is_correct"] is True

