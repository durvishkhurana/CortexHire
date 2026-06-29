from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "offline" / "04_train_student.py"
    spec = importlib.util.spec_from_file_location("train_student", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_select_winner_keeps_pointwise_when_harness_wins(tmp_path):
    mod = _load_train_module()
    model = tmp_path / "model"
    model.mkdir()
    (model / "meta.json").write_text('{"head": "pointwise"}', encoding="utf-8")
    lr = tmp_path / "model_lambdarank"
    lr.mkdir()
    (lr / "meta.json").write_text('{"head": "lambdarank"}', encoding="utf-8")

    mod.select_winner(
        str(tmp_path),
        {
            "winner": "pointwise",
            "metrics": {
                "pointwise": {"composite": 0.9},
                "lambdarank": {"composite": 0.8},
            },
        },
    )

    assert json.loads((model / "meta.json").read_text())["head"] == "pointwise"
    selection = json.loads((model / "selection.json").read_text())
    assert selection["selected_head"] == "pointwise"
