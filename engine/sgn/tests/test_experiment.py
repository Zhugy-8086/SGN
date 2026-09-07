# test_experiment.py - 训练实验框架测试（基础设施 A1）
import json
import os

import pytest

from engine.sgn.experiment import (
    run_seeds, aggregate, export_json, export_csv,
    save_ref_json, load_ref_json,
)


def fake_task(seed):
    """确定性伪任务：指标可预测，验证聚合口径。"""
    return {"loss": 1.0 / (seed + 1), "steps": seed * 10}


def test_run_seeds_collects():
    runs = run_seeds(fake_task, [1, 2, 3], label="fake")
    assert set(runs) == {1, 2, 3}
    assert runs[1] == {"loss": 0.5, "steps": 10}
    assert runs[3]["steps"] == 30


def test_aggregate_stats():
    runs = {1: {"loss": 1.0, "steps": 10},
            2: {"loss": 3.0, "steps": 20}}
    agg = aggregate(runs)
    st = agg["stats"]
    assert st["loss"]["mean"] == 2.0
    assert st["loss"]["std"] == pytest.approx(1.0)
    assert st["loss"]["min"] == 1.0 and st["loss"]["max"] == 3.0
    assert st["steps"]["mean"] == 15.0
    assert agg["per_seed"][1] == runs[1]


def test_aggregate_explicit_metrics():
    runs = {1: {"loss": 1.0, "extra": "x"}}
    agg = aggregate(runs, metrics=["loss"])
    assert agg["metrics"] == ["loss"]          # extra（非数值）不进入 stats
    assert "loss" in agg["stats"]


def test_aggregate_empty():
    assert aggregate({}) == {"per_seed": {}, "stats": {}}


def test_export_json_roundtrip(tmp_path):
    runs = {1: {"loss": 0.5}, 2: {"loss": 0.3}}
    path = export_json(runs, os.path.join(str(tmp_path), "r.json"), meta={"T": 100})
    with open(path) as f:
        d = json.load(f)
    assert d["meta"] == {"T": 100}
    assert d["runs"]["1"] == {"loss": 0.5}


def test_export_csv(tmp_path):
    runs = {1: {"loss": 0.5, "steps": 10}, 2: {"loss": 0.3, "steps": 20}}
    path = export_csv(runs, os.path.join(str(tmp_path), "r.csv"))
    with open(path) as f:
        lines = f.read().strip().splitlines()
    assert lines[0] == "seed,loss,steps"
    assert lines[1] == "1,0.5,10"
    assert lines[2] == "2,0.3,20"


def test_ref_json_roundtrip(tmp_path):
    """参照传递：格式对齐 resnet8_free_roam 的 free_roam_ref.json。"""
    path = os.path.join(str(tmp_path), "free_roam_ref.json")
    save_ref_json(path, T=1500, seeds=[7, 11], last50={7: 6.7e-6, 11: 8.5e-4})
    ref = load_ref_json(path)
    assert ref["T"] == 1500
    assert ref["seeds"] == [7, 11]
    assert ref["last50"] == {7: 6.7e-6, 11: 8.5e-4}
    # 与 resnet8_free_roam 原格式字段名兼容（free_anneal_last50）
    with open(path) as f:
        raw = json.load(f)
    assert raw["free_anneal_last50"] == {"7": 6.7e-6, "11": 8.5e-4}


def test_load_ref_json_missing_strategy(tmp_path):
    path = os.path.join(str(tmp_path), "ref.json")
    save_ref_json(path, T=1, seeds=[1], last50={1: 0.1}, strategy="free_only")
    ref = load_ref_json(path, strategy="free_anneal")   # 缺失 → last50 空
    assert ref["last50"] == {}
