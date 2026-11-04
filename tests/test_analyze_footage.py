"""Tests for the analyze_footage command's prompt configuration handling."""

import pytest

from scripts.analyze_footage import load_prompts_config


def test_load_prompts_config_rejects_invalid_json(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse .*prompts.json"):
        load_prompts_config(tmp_path)


def test_load_prompts_config_rejects_non_object_json(tmp_path):
    (tmp_path / "prompts.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_prompts_config(tmp_path)
