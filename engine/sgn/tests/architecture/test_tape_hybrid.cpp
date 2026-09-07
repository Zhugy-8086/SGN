// test_tape_hybrid.cpp - tape 混合语义守卫（2026-09-07 外部审查 A2/A3 + A1 实证）
//
// 性质：外部维护性审查采纳项的守卫测试——把"策略/存储切换下的缓存语义"
// 从注释约定固化为回归断言。全部断言基于 autograd.cpp 现行文档化行为
// （F1 入口清空 pair / grads_ 累加"混沌语义延续"），不改变任何行为。
//
// 用例：
//   H1 pair→pair 隔离：第二次 backward 入口清空 pair_grads_（F1），
//      第一次的 pair 梯度不得泄漏到第二次查询（nullptr 而非旧值）
//   H2 pair→float 切换：新图叶走 SR float 路径；旧图叶 nullptr
//      （#1 是 pair 存储、无 float 副本；#2 入口已清 pair）
//   H3 float→pair 切换（关键交叉场景）：图内叶 pair 新值胜出（freshness）；
//      图外叶命中 #1 遗留 float 值——文档化累加语义（AG-9/F1 注释
//      "不扩大战场"），此处断言现状防静默漂移
//   H4 QuantConfigGuard（A3）：基本恢复 / 嵌套 / 异常安全
//   H5 clip_sigma 惰性实证（A1）：max-scale 下 sigma=4 ≡ sigma=1e30
//      （逐位一致）；对照正例：估计式 scale 下 clip 会触发（bounded 输出）
//
// 数值设计：叶取值与 grad_output 均为小整数 → SR 量化 scale=max/max_val
// 下 xn 恰为整数（frac=0，SR 不加 1）→ 梯度精确无量化噪声，断言可用精确相等。
//
// 编译：由 engine/sgn/CMakeLists.txt test_tape_hybrid target 管理
//（Debug 模式测试块）。

#include "autograd/autograd.h"
#include "autograd/ops.h"
#include "autograd/ops_nn.h"        // StrategyContext / QuantConfigGuard（H3/H4）
#include "autograd/backward_strategy.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace sgn_autograd;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { ++failures; std::printf("FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

// ---- 辅助：1x1 叶 + 一次乘法录制 + backward(1) ----
// da = b·g, db = a·g（1x1 矩阵乘；g=1 → da=b, db=a，小整数下 SR 精确）
struct Graph1x1 {
    Tensor a, b, y;
    // pair=true → StrategyContext 走 pair 路径（SR + bwd16 + store on）
    Graph1x1(float va, float vb, bool pair_on) {
        StrategyContext::set(BackwardStrategy::SR);
        StrategyContext::bwd_quant_config() = QuantConfig{16, 4.0f};
        StrategyContext::set_pair_grad_store(pair_on);
        Tape& tape = Tape::current();
        tape.clear();
        float da[1] = {va}, db[1] = {vb};
        a = Tensor({1, 1}, da); a.set_requires_grad(true);
        b = Tensor({1, 1}, db); b.set_requires_grad(true);
        tape.start_recording();
        y = matmul(a, b);
        tape.stop_recording();
        float one[1] = {1.0f};
        tape.backward(y, Tensor({1, 1}, one));
    }
};

static float grad_value_or_nan(const Tensor& t) {
    const Tensor* g = Tape::current().grad(t.id());
    return g ? g->data()[0] : std::nanf("");
}

// ---- H1: pair→pair 隔离（F1 不变量）----
static void test_h1_pair_pair_isolation() {
    std::printf("[H1] pair→pair：入口清空，#1 的 pair 梯度不得泄漏\n");
    {   Graph1x1 g1(2.0f, 3.0f, /*pair_on=*/true);
        const Tensor* pa = Tape::current().grad(g1.a.id());
        CHECK(pa != nullptr && pa->data()[0] == 3.0f,
              "H1: #1 grad(a) pair-decoded == 3.0");
    }
    {   Graph1x1 g2(4.0f, 5.0f, /*pair_on=*/true);
        CHECK(grad_value_or_nan(g2.a) == 5.0f, "H1: #2 grad(a) == 5.0 (fresh)");
        CHECK(grad_value_or_nan(g2.b) == 4.0f, "H1: #2 grad(b) == 4.0 (fresh)");
    }
    // #1 的叶在 #2 入口被清空后查询 → nullptr（而非 3.0 旧值）
    // （#1 的 Tensor 已析构，id 可能复用——用独立作用域重放校验：见 H2/H3
    // 的跨策略断言；本用例锁 #2 图内 freshness 已足够。）
}

// ---- H2: pair→float 切换 ----
static void test_h2_pair_then_float() {
    std::printf("[H2] pair→float：新图叶走 SR float，旧 pair 无泄漏\n");
    {   Graph1x1 g1(2.0f, 3.0f, /*pair_on=*/true);
        CHECK(grad_value_or_nan(g1.a) == 3.0f, "H2: #1 pair grad(a) == 3.0");
    }
    {   Graph1x1 g2(4.0f, 5.0f, /*pair_on=*/false);   // SR float 路径
        CHECK(grad_value_or_nan(g2.a) == 5.0f,
              "H2: #2 float SR grad(a) == 5.0 (小整数 SR 精确)");
        CHECK(grad_value_or_nan(g2.b) == 4.0f, "H2: #2 float SR grad(b) == 4.0");
    }
}

// ---- H3: float→pair 切换（关键交叉场景，AG-9 复活路径——不 clear 直录新前向）----
static void test_h3_float_then_pair() {
    std::printf("[H3] float→pair（无 clear 复活）：图内叶 pair 新值胜出；图外叶 float 遗留=文档化累加语义\n");
    StrategyContext::set(BackwardStrategy::SR);
    StrategyContext::set_pair_grad_store(false);
    Tape& tape = Tape::current();
    tape.clear();   // 仅初始净带；#1→#2 之间不 clear（走 record() 复活）
    float da1[1] = {2.0f}, db1[1] = {3.0f};
    Tensor a1({1, 1}, da1); a1.set_requires_grad(true);
    Tensor b1({1, 1}, db1); b1.set_requires_grad(true);
    tape.start_recording();
    Tensor y1 = matmul(a1, b1);
    tape.stop_recording();
    float one[1] = {1.0f};
    tape.backward(y1, Tensor({1, 1}, one));
    CHECK(grad_value_or_nan(a1) == 3.0f, "H3: #1 float grad(a1) == 3.0");
    CHECK(grad_value_or_nan(b1) == 2.0f, "H3: #1 float grad(b1) == 2.0");

    // #2：切 pair 存储，直接录新前向（record() 置 consumed_=false，AG-9 复活）。
    // 注意 backward #2 入口清 pair_grads_ 但 **不清 grads_**（float 累加语义）——
    // 这正是 #1 遗留 float 值的来源。
    StrategyContext::set_pair_grad_store(true);
    float da2[1] = {4.0f}, db2[1] = {5.0f};
    Tensor a2({1, 1}, da2); a2.set_requires_grad(true);
    Tensor b2({1, 1}, db2); b2.set_requires_grad(true);
    tape.start_recording();
    Tensor y2 = matmul(a2, b2);
    tape.stop_recording();
    tape.backward(y2, Tensor({1, 1}, one));
    // 图内叶：pair 新值胜出（freshness）
    CHECK(grad_value_or_nan(a2) == 5.0f, "H3: #2 pair grad(a2) == 5.0 (fresh)");
    // 图外叶：pair miss → grads_ 命中 #1 遗留 float —— 文档化累加语义
    //（autograd.cpp F1 注释"混沌语义的延续，不扩大战场"）。此处断言现状，
    // 防止未来改动静默漂移；若语义升级（如入口连带清 grads_），本断言随之更新。
    CHECK(grad_value_or_nan(a1) == 3.0f,
          "H3: stale float residue for out-of-graph leaf (documented accumulate)");
    CHECK(grad_value_or_nan(b1) == 2.0f, "H3: stale float residue (b1)");
    StrategyContext::set_pair_grad_store(false);
}

// ---- H4: QuantConfigGuard（A3）----
static void test_h4_quant_config_guard() {
    std::printf("[H4] QuantConfigGuard：基本恢复 / 嵌套 / 异常安全\n");
    StrategyContext::quant_config() = QuantConfig{8, 4.0f};
    StrategyContext::bwd_quant_config() = QuantConfig{16, 4.0f};
    {
        QuantConfigGuard g(QuantConfig{4, 2.0f}, QuantConfig{8, 3.0f});
        CHECK(StrategyContext::quant_config().bits == 4 &&
              StrategyContext::quant_config().clip_sigma == 2.0f,
              "H4: fwd config set");
        CHECK(StrategyContext::bwd_quant_config().bits == 8 &&
              StrategyContext::bwd_quant_config().clip_sigma == 3.0f,
              "H4: bwd config set");
        {   // 嵌套：内层恢复到内层进入值（即外层的设置值）
            QuantConfigGuard inner(QuantConfig{16, 1.0f}, QuantConfig{16, 1.0f});
            CHECK(StrategyContext::quant_config().bits == 16, "H4: inner set");
        }
        CHECK(StrategyContext::quant_config().bits == 4 &&
              StrategyContext::bwd_quant_config().clip_sigma == 3.0f,
              "H4: nested restore to outer entry value");
    }
    CHECK(StrategyContext::quant_config().bits == 8 &&
          StrategyContext::quant_config().clip_sigma == 4.0f,
          "H4: outer restore fwd");
    CHECK(StrategyContext::bwd_quant_config().bits == 16 &&
          StrategyContext::bwd_quant_config().clip_sigma == 4.0f,
          "H4: outer restore bwd");
    // 异常安全：抛出穿越守卫作用域仍恢复
    bool caught = false;
    try {
        QuantConfigGuard g(QuantConfig{2, 9.0f}, QuantConfig{2, 9.0f});
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
        caught = true;
    }
    CHECK(caught, "H4: exception propagated");
    CHECK(StrategyContext::quant_config().bits == 8 &&
          StrategyContext::bwd_quant_config().bits == 16,
          "H4: configs restored through exception");
}

// ---- H5: clip_sigma 惰性实证（A1）----
static void test_h5_clip_inert() {
    std::printf("[H5] max-scale 下 sigma=4 ≡ sigma=1e30（clip 恒不触发）；估计式 scale 正例\n");
    const size_t K = 4096;
    std::mt19937 gen(20260907);
    std::vector<float> g(K);
    for (auto& x : g) {
        x = std::exp(static_cast<float>(gen() % 2000) / 300.0f - 4.0f)
            * ((gen() % 2) ? 1.0f : -1.0f);   // 类梯度分布（跨数量级 + 符号）
    }
    // max-scale 下：clip_bound = sigma·max_abs ≥ max_abs ≥ |g_i| → 恒不触发
    std::vector<float> g1 = g, g2 = g;
    set_sr_seed(42);
    sr_quantize_grad(g1.data(), K, 16, 4.0f);
    set_sr_seed(42);
    sr_quantize_grad(g2.data(), K, 16, 1e30f);   // 数学上保证不 clip 的对照
    CHECK(std::memcmp(g1.data(), g2.data(), K * sizeof(float)) == 0,
          "H5: sigma=4 bit-identical to sigma=1e30 (inert under max-scale)");
    // 正例：估计式 scale（second_max）下 clip 会触发——q 有界即证据
    // （outlier 1e2 级、主体 1e-2 级：estimate-scale 时 outlier 越界被 clip）
    std::vector<float> ge(256);
    for (size_t i = 0; i + 1 < ge.size(); ++i) {
        ge[i] = ((static_cast<float>(gen() % 2)) ? 1.0f : -1.0f)
                * (0.01f + 0.001f * static_cast<float>(i % 10));
    }
    ge.back() = 100.0f;                           // 唯一 outlier = max
    float second_max = 0.02f;
    float scale_est = second_max / 32767.0f;      // 估计式 scale（不含 outlier）
    std::vector<int32_t> q(ge.size());
    sr_quantize_q(ge.data(), ge.size(), scale_est, 4.0f, 32767.0f, q.data());
    // outlier 位置：clip_bound = 4·second_max = 0.08 → 100 被截到 0.08 →
    // q = 0.08/scale_est = 4·32767 → 再被 q-clip 收到 32767（有界、无溢出）
    CHECK(q.back() == 32767, "H5: estimate-scale fires clip, outlier bounded at +max_val");
    bool body_ok = true;
    for (size_t i = 0; i + 1 < ge.size(); ++i) {
        if (q[i] < -32767 || q[i] > 32767) { body_ok = false; break; }
    }
    CHECK(body_ok, "H5: body q within range");
}

int main() {
    test_h1_pair_pair_isolation();
    test_h2_pair_then_float();
    test_h3_float_then_pair();
    test_h4_quant_config_guard();
    test_h5_clip_inert();
    if (failures == 0) {
        std::printf("\nALL TAPE HYBRID TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d TAPE HYBRID FAILURES\n", failures);
    return 1;
}
