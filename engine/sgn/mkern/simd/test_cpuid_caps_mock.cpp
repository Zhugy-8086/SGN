// test_cpuid_caps_mock.cpp - CPUID caps 纯派生 mock 回归（2026-09-07 外部审查 A4）
//
// 目的：把 simd_dispatch.cpp caps 派生逻辑与真机解耦——直接喂寄存器断言，
// 固化历史 bug（Arrow Lake OSPKE leaf 误读，2026-09-02 修正）为回归测试。
// 新增 ISA 位检测时必须在本文件同步新增 mock 用例（对应 simd_dispatch.cpp
// caps_from_registers 注释的纪律条款）。
//
// 接缝说明：caps_from_registers 是匿名命名空间 static——本文件直接
// #include 实现以触达（标准 static 测试手法；不污染 simd_api.h 平台无关
// 契约，不新增导出符号）。CMake 目标因此不再单列 simd_dispatch.cpp。
//
// 编译：由 engine/sgn/CMakeLists.txt test_cpuid_caps_mock target 管理
//（Debug 模式测试块，与 test_pair_grad_carrier 同批）。

#include "simd_dispatch.cpp"

#include <cstdio>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { ++failures; std::printf("FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

// 注：simd_dispatch.cpp 的 CpuCaps / caps_from_registers 位于 sgn::simd 的
// 匿名命名空间（文件级 namespace sgn::simd 内层 namespace {}），include 后
// 经 using-directive 直呼可用。
using namespace sgn::simd;

// ---- 寄存器构造辅助 ----
struct Regs {
    int l1[4]  = {0, 0, 0, 0};
    int l7_0[4] = {0, 0, 0, 0};
    int l7_1[4] = {0, 0, 0, 0};
    unsigned long long xcr0 = 0;
    void set_avx_gate() {          // leaf1 ECX: XSAVE=27, AVX=28, SSSE3=9
        l1[2] |= (1 << 27) | (1 << 28) | (1 << 9);
    }
    void set_avx2()      { l7_0[1] |= (1 << 5); }    // leaf7 sub0 EBX bit5
    void set_avx512f()   { l7_0[1] |= (1 << 16); }   // leaf7 sub0 EBX bit16
    void set_avx512bw()  { l7_0[1] |= (1 << 30); }   // leaf7 sub0 EBX bit30
    void set_ospke()     { l7_0[2] |= (1 << 4); }    // leaf7 sub0 ECX bit4（旧 bug 误读位）
    void set_avx512vnni(){ l7_0[2] |= (1 << 11); }   // leaf7 sub0 ECX bit11
    void set_avxvnni()   { l7_1[0] |= (1 << 4); }    // leaf7 sub1 EAX bit4（正确位）
};

static CpuCaps derive(const Regs& r) {
    return caps_from_registers(r.l1, r.l7_0, r.l7_1, r.xcr0);
}

// ---- M1: Arrow Lake OSPKE 误读 bug 回归（2026-09-02 修正固化）----
// 场景：OSPKE=0（Arrow Lake/Windows 实测）但 CPU 真有 AVX-VNNI。
// 旧代码读 sub0 ECX[4]=OSPKE 位 → avx_vnni 永远 false → dot8/dot4 静默落标量；
// EPYC/Linux 因 OSPKE=1 侥幸误判掩盖。修正后读 sub1 EAX[4]。
static void test_m1_arrow_lake_regression() {
    std::printf("[M1] Arrow Lake 回归：OSPKE=0 + 真实 AVX-VNNI → 必须检出\n");
    Regs r;
    r.set_avx_gate();
    r.set_avx2();
    r.set_avxvnni();       // 真实硬件能力位（sub1 EAX bit4）
    r.xcr0 = 0x6;          // XMM+YMM 使能
    // 注意：OSPKE 位（sub0 ECX bit4）故意保持 0
    CpuCaps c = derive(r);
    CHECK(c.avx_vnni == true,  "M1: avx_vnni must be detected via leaf7 sub1 EAX[4]");
    CHECK(c.avx2 == true,      "M1: avx2");
    CHECK(c.ssse3 == true,     "M1: ssse3");
    CHECK(c.avx512f == false,  "M1: no avx512f");
}

// ---- M2: 无 AVX-VNNI + OSPKE=0 → 正确回退（avx2 承接 dot8）----
static void test_m2_no_vnni_fallback() {
    std::printf("[M2] 无 VNNI（OSPKE=0）→ avx_vnni=false，回退 avx2 前提\n");
    Regs r;
    r.set_avx_gate();
    r.set_avx2();
    r.set_ospke();         // OSPKE=1（EPYC 场景对照）
    r.xcr0 = 0x6;
    CpuCaps c = derive(r);
    CHECK(c.avx_vnni == false, "M2: no vnni bit → false");
    CHECK(c.avx2 == true,      "M2: avx2 fallback ready");
}

// ---- M3: OS 未使能 YMM（xcr0=0x2 仅 XMM）→ AVX 系全关，SSSE3 不受门控 ----
static void test_m3_xcr0_gate() {
    std::printf("[M3] xcr0=0x2（无 YMM）→ AVX/AVX2/VNNI 全 false，ssse3 保持\n");
    Regs r;
    r.set_avx_gate();
    r.set_avx2();
    r.set_avxvnni();
    r.xcr0 = 0x2;          // 仅 XMM state
    CpuCaps c = derive(r);
    CHECK(c.avx2 == false,     "M3: avx2 gated by xcr0");
    CHECK(c.avx_vnni == false, "M3: vnni gated");
    CHECK(c.ssse3 == true,     "M3: ssse3 not xcr0-gated");
}

// ---- M4: AVX-512 全家福（xcr0=0xE6：opmask + ZMM hi 使能）----
static void test_m4_avx512_full() {
    std::printf("[M4] xcr0=0xE6 + F/BW/VNNI 位全在 → 三位全 true\n");
    Regs r;
    r.set_avx_gate();
    r.set_avx2();
    r.set_avx512f();
    r.set_avx512bw();
    r.set_avx512vnni();
    r.set_avxvnni();
    r.xcr0 = 0xE6;
    CpuCaps c = derive(r);
    CHECK(c.avx512f == true,   "M4: avx512f");
    CHECK(c.avx512_bw == true, "M4: avx512bw");
    CHECK(c.avx512_vnni == true, "M4: avx512vnni");
    CHECK(c.avx_vnni == true,  "M4: avx_vnni");
}

// ---- M5: AVX-512 但 OS 未使能 ZMM（xcr0=0x6）→ f/bw 强制 false ----
// 现状语义（派生逐行原样搬移）：avx512_vnni 位不受 xcr0-E6 门控（仅 f/bw
// 被 172-180 行联合清位）——与实现一致，文档化于此；后端选择侧由
// simd_backend() 的 f&&bw&&vnni 联合判定兜底。
static void test_m5_avx512_no_zmm_state() {
    std::printf("[M5] xcr0=0x6 + F/BW 位在 → f/bw 强制 false（vnni 位照读，现状语义）\n");
    Regs r;
    r.set_avx_gate();
    r.set_avx2();
    r.set_avx512f();
    r.set_avx512bw();
    r.set_avx512vnni();
    r.xcr0 = 0x6;
    CpuCaps c = derive(r);
    CHECK(c.avx512f == false,  "M5: avx512f forced false");
    CHECK(c.avx512_bw == false, "M5: avx512bw forced false");
    CHECK(c.avx2 == true,      "M5: avx2 still available");
}

// ---- M6: AVX 硬件无但 SSSE3 有 → 仅 ssse3（门未开路径，l7 全零输入合法）----
static void test_m6_gate_closed() {
    std::printf("[M6] 无 AVX/XSAVE 门 → 仅 ssse3（l7 寄存器全零合法输入）\n");
    Regs r;
    r.l1[2] |= (1 << 9);   // 仅 SSSE3
    r.set_avx2();          // 即使 leaf7 声称有 AVX2 也必须被门挡住
    r.xcr0 = 0;
    CpuCaps c = derive(r);
    CHECK(c.ssse3 == true,  "M6: ssse3");
    CHECK(c.avx2 == false,  "M6: gate closed → avx2 false");
    CHECK(c.avx_vnni == false, "M6: gate closed → vnni false");
}

int main() {
    test_m1_arrow_lake_regression();
    test_m2_no_vnni_fallback();
    test_m3_xcr0_gate();
    test_m4_avx512_full();
    test_m5_avx512_no_zmm_state();
    test_m6_gate_closed();
    if (failures == 0) {
        std::printf("\nALL CPUID MOCK TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d MOCK TEST FAILURES\n", failures);
    return 1;
}
