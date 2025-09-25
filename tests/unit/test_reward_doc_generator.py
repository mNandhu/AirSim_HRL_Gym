from pathlib import Path

from utils.reward_doc import generate_reward_contract


def test_generate_reward_contract_creates_file(tmp_path: Path):
    output = tmp_path / "reward.md"
    generated = generate_reward_contract(output)

    assert generated.exists()
    content = generated.read_text(encoding="utf-8")
    assert "Reward Contract" in content
    assert "collision" in content
