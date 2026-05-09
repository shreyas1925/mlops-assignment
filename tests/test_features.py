from __future__ import annotations

from features import build_preprocessor


def test_build_preprocessor_has_two_transformer_groups():
    transformer = build_preprocessor()
    names = [item[0] for item in transformer.transformers]
    assert names == ["num", "cat"]

