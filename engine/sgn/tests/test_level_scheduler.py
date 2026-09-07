# test_level_scheduler.py - LevelScheduler Python 接口完备性核查（基础设施 A3）
# 覆盖：构造/策略管理/查询/自适应链/序列化往返/NeuronLevelStats/建议接口
import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[2]   # engine/
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from sgn.level import (  # noqa: E402
    LevelScheduler, StandardStrategy, AdaptiveStrategy, NeuronLevelStats,
    LevelOperation, LevelConstants,
)


def test_constructor_defaults():
    s = LevelScheduler()
    assert s.get_strategy("standard") is not None
    assert s.get_strategy("adaptive") is not None
    # 默认策略 standard(level=0)：未绑定神经元 → 默认 level 0
    assert s.get_level(999) == 0
    assert s.step_counter == 0


def test_strategy_management():
    s = LevelScheduler()
    s.register_strategy("std3", StandardStrategy(3))
    assert s.get_strategy("std3") is not None
    assert s.get_strategy("std3").default_level() == 3
    s.bind_neuron(1, "std3")
    assert s.get_level(1) == 3                      # 固定 level
    assert s.get_level(1, LevelOperation.ADD) == 3  # operation 被 standard 忽略
    # 未知策略绑定抛异常
    try:
        s.bind_neuron(2, "nope")
        assert False, "应抛异常"
    except RuntimeError:
        pass


def test_set_default_strategy():
    s = LevelScheduler()
    s.register_strategy("ad", AdaptiveStrategy(base_level=1))
    s.set_default_strategy("ad")
    # 未绑定神经元 → 走默认 adaptive：无 stats 时返回 base_level=1
    assert s.get_level(50) == 1
    try:
        s.set_default_strategy("missing")
        assert False, "应抛异常"
    except RuntimeError:
        pass


def test_adaptive_get_level_uses_stats():
    s = LevelScheduler(adapt_interval=1000)
    s.register_strategy("ad", AdaptiveStrategy(base_level=0))
    s.bind_neuron(3, "ad")
    # bind 创建 stats（current_level=LEVEL_DEFAULT=0），get_level 返回 stats.current_level
    assert s.get_level(3) == 0
    stats = s.get_stats(3)
    assert stats is not None and stats.neuron_id == 3


def test_update_stats_counts():
    s = LevelScheduler(adapt_interval=1000)   # 长间隔避免自适应干扰
    s.bind_neuron(0, "standard")
    s.update_stats(0, match=5, verified=True)
    s.update_stats(0, match=7, verified=False)
    st = s.get_stats(0)
    assert st.total_count == 2
    assert st.verified_count == 1
    assert st.last_match == 7
    assert s.step_counter == 2               # scheduler.step_counter（update 计数）


def test_adaptive_promotion_chain():
    """自适应链：match 恒定（MAD=0 < 阈值/4）→ 每 adapt_interval 升一级。

    注：升 level 受 peak_level 硬上限约束（v5.1.9-fix P1-2）——bind 创建 stats 时
    peak=current=0，须先预置 peak_level 允许升级。
    """
    s = LevelScheduler(adapt_interval=3)
    s.register_strategy("ad", AdaptiveStrategy(
        base_level=0, variance_threshold=100.0, history_window=3,
        demotion_min_samples=5))
    s.bind_neuron(0, "ad")
    s.get_stats(0).peak_level = 2           # 允许升到 2（LEVEL_MAX 封顶）
    assert s.get_level(0) == 0
    # 3 次 update（step_counter=3 触发 check_adaptation）
    for _ in range(3):
        s.update_stats(0, match=5, verified=True)
    st = s.get_stats(0)
    assert st.current_level == 1              # MAD=0 → 升一级
    assert st.level_change_count == 1
    assert s.get_level(0) == 1
    # 再 3 次 → MAD 仍 <25 → 再升到 2（LEVEL_MAX=2 封顶）
    for _ in range(3):
        s.update_stats(0, match=5, verified=True)
    assert s.get_stats(0).current_level == 2
    assert s.get_stats(0).level_change_count == 2


def test_adaptive_demotion_chain():
    """自适应链：验证率低 + 样本足 → 主动降级。"""
    s = LevelScheduler(adapt_interval=1)
    s.register_strategy("ad", AdaptiveStrategy(
        base_level=1, variance_threshold=100.0, history_window=1,
        demotion_verification_threshold=0.5, demotion_min_samples=4))
    s.bind_neuron(0, "ad")
    st = s.get_stats(0)
    st.current_level = 1                      # 预置非最粗粒度
    # feed 4 次，verified 低（rate=0.25 < 0.5）→ 降级
    for _ in range(4):
        s.update_stats(0, match=50, verified=False)
    assert s.get_stats(0).current_level < 1   # 0（LEVEL_MIN 之上）


def test_suggest_adaptation_insufficient_samples():
    """样本不足（< history_window）→ 无建议。"""
    strat = AdaptiveStrategy(base_level=0, variance_threshold=100.0, history_window=50)
    st = NeuronLevelStats(0)
    for _ in range(10):
        st.update(5, True)
    assert strat.suggest_adaptation(st) is None


def test_suggest_demotion_thresholds():
    """验证率低于阈值且样本足 → 降一级；已最粗 → 不降。"""
    strat = AdaptiveStrategy(base_level=1, demotion_verification_threshold=0.5,
                             demotion_min_samples=5)
    st = NeuronLevelStats(0)
    st.current_level = 1
    for _ in range(5):
        st.update(5, verified=False)          # rate=0 < 0.5
    assert strat.suggest_demotion(st) == 0
    st2 = NeuronLevelStats(0)
    st2.current_level = LevelConstants.LEVEL_MIN   # 已最粗
    for _ in range(5):
        st2.update(5, verified=False)
    assert strat.suggest_demotion(st2) is None


def test_neuron_stats_math():
    st = NeuronLevelStats(0)
    assert st.match_variance() == 0           # <2 样本
    for m in (10, 10, 10):
        st.update(m, True)
    assert st.match_variance() == 0           # 恒定 → MAD=0
    assert st.verification_rate() == 1.0
    for m in (0, 20):
        st.update(m, True)                    # MAD = (10*4 + 10 + 10)/6 = 10... 见断言
    # 方案 D：grad_variance 累积
    st.update(0, True, grad_variance=1.0)
    st.update(0, True, grad_variance=3.0)
    assert st.grad_variance == 2.0            # (1+3)/2


def test_serialization_roundtrip():
    """序列化往返：step_counter / 神经元绑定 / 统计恢复。

    注：自定义策略不参与序列化（策略是代码非数据，to_dict 只存绑定名）——
    from_dict 重建后自定义策略名回退默认。此处用内置 adaptive 策略验证可恢复。
    """
    s = LevelScheduler(adapt_interval=3)
    s.register_strategy("ad", AdaptiveStrategy(base_level=0))
    s.bind_neuron(1, "standard")
    s.bind_neuron(2, "ad")
    s.update_stats(2, match=5, verified=True)
    s.update_stats(2, match=5, verified=True)
    s.update_stats(2, match=5, verified=True)  # step_counter=3 → 触发自适应（但 peak=0 挡住升级）
    d = s.to_dict()
    assert "step_counter" in d
    restored = LevelScheduler.from_dict(d)
    assert restored.step_counter == s.step_counter
    assert restored.get_level(1) == s.get_level(1)   # 内置 standard 绑定恢复
    assert restored.get_stats(2) is not None
    assert restored.get_stats(2).total_count == s.get_stats(2).total_count


def test_custom_strategy_binding_not_serialized():
    """已知行为：自定义策略不序列化，from_dict 后绑定回退默认策略。"""
    s = LevelScheduler()
    s.register_strategy("std5", StandardStrategy(5))
    s.bind_neuron(1, "std5")
    d = s.to_dict()
    restored = LevelScheduler.from_dict(d)
    assert restored.get_level(1) == 0   # std5 缺失 → 回退默认 standard(0)


def test_serialization_corrupt_tolerant():
    """corrupt 输入不崩溃（审计 L1-2：逐字段 try/catch）。"""
    s = LevelScheduler.from_dict({"step_counter": "not_a_number"})
    assert s is not None
    assert s.step_counter == 0
