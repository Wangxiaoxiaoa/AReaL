import pytest
import torch
from torch import nn

from areal.engine.fsdp_engine import (
    _build_qwen_vl_rope_index_kwargs,
    _get_qwen_vl_get_rope_index,
)


class RopeIndexModel(nn.Module):
    def get_rope_index(self):
        return "rope-index"


class NestedRopeIndexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RopeIndexModel()


class MissingRopeIndexModel(nn.Module):
    pass


class Qwen3StyleModel(nn.Module):
    class Config:
        image_token_id = 101
        video_token_id = 102

    config = Config()

    def get_rope_index(self, input_ids, mm_token_type_ids, **kwargs):
        return input_ids, mm_token_type_ids


class Qwen2StyleModel(nn.Module):
    class Config:
        image_token_id = 101
        video_token_id = 102

    config = Config()

    def get_rope_index(self, input_ids, **kwargs):
        return input_ids, kwargs


def test_get_qwen_vl_get_rope_index_uses_top_level_method():
    get_rope_index = _get_qwen_vl_get_rope_index(RopeIndexModel())

    assert get_rope_index() == "rope-index"


def test_get_qwen_vl_get_rope_index_uses_inner_model_method():
    get_rope_index = _get_qwen_vl_get_rope_index(NestedRopeIndexModel())

    assert get_rope_index() == "rope-index"


def test_get_qwen_vl_get_rope_index_raises_clear_error():
    with pytest.raises(AttributeError, match="requires get_rope_index"):
        _get_qwen_vl_get_rope_index(MissingRopeIndexModel())


def test_build_qwen_vl_rope_index_kwargs_adds_mm_token_type_ids_when_required():
    model = Qwen3StyleModel()
    input_ids = torch.tensor([[1, 101, 101, 2, 102]])

    kwargs = _build_qwen_vl_rope_index_kwargs(
        get_rope_index=model.get_rope_index,
        model_config=model.config,
        input_ids=input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    )

    assert kwargs["mm_token_type_ids"].tolist() == [[0, 1, 1, 0, 2]]


def test_build_qwen_vl_rope_index_kwargs_skips_mm_token_type_ids_when_not_required():
    model = Qwen2StyleModel()

    kwargs = _build_qwen_vl_rope_index_kwargs(
        get_rope_index=model.get_rope_index,
        model_config=model.config,
        input_ids=torch.tensor([[1, 101]]),
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    )

    assert "mm_token_type_ids" not in kwargs
