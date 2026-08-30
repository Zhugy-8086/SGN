// sgn_benchmark.cpp - SGN SIMD 原语层完整测试：正确性验证 + 性能基准 + JSON 输出
//
// 编译（在 SGN 仓库根目录执行）：
//   g++ -O3 -std=c++23 -mavx2 -mavxvnni -mavx512f -mavx512bw -mavx512vl -mavx512vnni \
//       -I. sgn_benchmark/sgn_benchmark.cpp \
//       simd/scalar.cpp simd/simd_dispatch.cpp \
//       simd/x86/avx2.cpp simd/x86/avxvnni.cpp simd/x86/ssse3.cpp \
//       simd/x86/avx512.cpp simd/x86/avx512vnni.cpp simd/arm/neon.cpp \
//       -o sgn_benchmark/sgn_benchmark
//
// 运行：
//   ./sgn_benchmark/sgn_benchmark              # 输出人类可读 + result.json
//   ./sgn_benchmark/sgn_benchmark --json-only  # 仅输出 JSON
//
// 覆盖原语：dot16, dot8, dot4, sum_f32, sum_sq_dev_f32, sum_sumprod_f32,
//           accum_f32, decode_i16_f32, reverse_bytes8, batch_reverse_u8

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>
#include "simd/simd_api.h"

using namespace sgn::simd;

// ============================================================================
// 工具函数
// ============================================================================

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static std::mt19937_64 rng(20260831);

template<typename T>
static T rand_range(T lo, T hi) {
    return static_cast<T>(rng() % (uint64_t)(hi - lo + 1)) + lo;
}

// ============================================================================
// 标量锚点（用于正确性对比）
// ============================================================================

static int64_t ref_dot16(const int16_t* a, const int16_t* b, size_t K) {
    int64_t s = 0;
    for (size_t i = 0; i < K; i++) s += (int64_t)a[i] * b[i];
    return s;
}
static int64_t ref_dot8(const uint8_t* a, const int8_t* b, size_t K) {
    int64_t s = 0;
    for (size_t i = 0; i < K; i++) s += (int64_t)a[i] * b[i];
    return s;
}
static float ref_sum_f32(const float* p, int64_t n, int64_t stride) {
    float s = 0;
    for (int64_t i = 0; i < n; i++) s += p[i * stride];
    return s;
}
static float ref_sum_sq_dev(const float* p, int64_t n, int64_t stride, float mu) {
    float s = 0;
    for (int64_t i = 0; i < n; i++) { float d = p[i*stride]-mu; s += d*d; }
    return s;
}
static void ref_decode(const uint64_t* pv, int n, float scale, float* out) {
    for (int i = 0; i < n; i++)
        out[i] = (float)(int16_t)(pv[i] & 0xFFFF) * scale;
}
static void ref_reverse8(uint64_t packed, int n, int64_t* out) {
    for (int i = 0; i < n; i++)
        out[i] = (int64_t)((packed >> (8*(n-1-i))) & 0xFF);
}
static void ref_batch_reverse(const uint64_t* pv, int nv, int ns, int64_t* out) {
    for (int i = 0; i < nv; i++) {
        const uint8_t* bytes = (const uint8_t*)&pv[i];
        for (int j = 0; j < ns; j++) out[i*ns+j] = bytes[ns-1-j];
    }
}

// ============================================================================
// 正确性测试
// ============================================================================

struct TestResult {
    std::string name;
    int total = 0;
    int passed = 0;
    std::string detail;
};

static TestResult test_dot16() {
    TestResult r{"dot16", 0, 0, ""};
    size_t sizes[] = {0,1,7,8,15,16,17,31,32,33,63,64,65,127,128,1000,65536};
    for (size_t K : sizes) {
        std::vector<int16_t> a(K), b(K);
        for (size_t i = 0; i < K; i++) {
            a[i] = (i%3==0) ? INT16_MIN : rand_range<int16_t>(-30000,30000);
            b[i] = (i%5==0) ? INT16_MIN : rand_range<int16_t>(-30000,30000);
        }
        r.total++;
        int64_t got = dot16(a.data(), b.data(), K);
        int64_t exp = ref_dot16(a.data(), b.data(), K);
        if (got == exp) r.passed++;
        else if (r.detail.empty()) {
            char buf[128]; snprintf(buf,sizeof(buf),"K=%zu got=%lld exp=%lld",K,(long long)got,(long long)exp);
            r.detail = buf;
        }
    }
    // 全 -32768 极值
    {
        std::vector<int16_t> a(64, INT16_MIN), b(64, INT16_MIN);
        r.total++;
        if (dot16(a.data(),b.data(),64) == ref_dot16(a.data(),b.data(),64)) r.passed++;
    }
    return r;
}

static TestResult test_dot8_dot4() {
    TestResult r{"dot8/dot4", 0, 0, ""};
    size_t sizes[] = {0,1,7,8,31,32,33,63,64,65,127,128,1000,65536};
    for (size_t K : sizes) {
        std::vector<uint8_t> u(K);
        std::vector<int8_t> s(K);
        for (size_t i = 0; i < K; i++) {
            u[i] = (i%4==0) ? 255 : (uint8_t)rng();
            s[i] = (i%6==0) ? 127 : (i%7==0 ? -128 : (int8_t)rng());
        }
        r.total++;
        if (dot8(u.data(),s.data(),K) == ref_dot8(u.data(),s.data(),K)) r.passed++;
        else if (r.detail.empty()) { char buf[64]; snprintf(buf,sizeof(buf),"dot8 K=%zu",K); r.detail=buf; }
        r.total++;
        if (dot4(u.data(),s.data(),K) == ref_dot8(u.data(),s.data(),K)) r.passed++;
        else if (r.detail.empty()) { char buf[64]; snprintf(buf,sizeof(buf),"dot4 K=%zu",K); r.detail=buf; }
    }
    return r;
}

static TestResult test_sum_series() {
    TestResult r{"sum_f32 系", 0, 0, ""};
    int64_t sizes[] = {0,1,7,8,9,15,16,17,31,32,100,1000};
    for (int64_t n : sizes) {
        for (int64_t stride : {1, 3, 5}) {
            int64_t total = (n==0)?0:(n-1)*stride+1;
            std::vector<float> p(total), q(total);
            for (auto& x : p) x = (float)(rng()%2001)-1000.0f;
            for (auto& x : q) x = (float)(rng()%101)-50.0f;
            // sum 系为 kRounding：SIMD 分块累加与标量顺序累加的浮点舍入顺序不同，
            // 属预期差异（同 avx2.cpp/avx512.cpp 注释），用相对容差比较而非精确 ==。
            auto feq = [](float a, float b) {
                float d = std::fabs(a-b);
                float m = std::max(std::fabs(a), std::fabs(b));
                return d < 1e-3f || (m > 0 && d/m < 1e-4f);
            };
            r.total++;
            if (feq(sum_f32(p.data(),n,stride), ref_sum_f32(p.data(),n,stride))) r.passed++;
            r.total++;
            if (feq(sum_sq_dev_f32(p.data(),n,stride,3.0f), ref_sum_sq_dev(p.data(),n,stride,3.0f))) r.passed++;
            float s1=0,sp1=0,s2=0,sp2=0;
            sum_sumprod_f32(p.data(),q.data(),n,stride,&s1,&sp1);
            s2=ref_sum_f32(p.data(),n,stride);
            for (int64_t i=0;i<n;i++) sp2+=p[i*stride]*q[i*stride];
            r.total++;
            if (feq(s1,s2) && feq(sp1,sp2)) r.passed++;
        }
    }
    return r;
}

static TestResult test_decode() {
    TestResult r{"decode_i16_f32", 0, 0, ""};
    int sizes[] = {0,1,3,7,8,9,15,16,100,1000};
    for (int n : sizes) {
        std::vector<uint64_t> pv(n);
        for (int i = 0; i < n; i++) {
            pv[i] = rng() & 0xFFFF;
            if (i%3==0) pv[i] = 0x8000;
            if (i%5==0) pv[i] = 0xFFFF;
        }
        std::vector<float> r1(n), r2(n);
        decode_i16_f32(pv.data(), n, 0.25f, r1.data());
        ref_decode(pv.data(), n, 0.25f, r2.data());
        r.total++;
        if (std::memcmp(r1.data(),r2.data(),n*sizeof(float))==0) r.passed++;
        else if (r.detail.empty()) { char buf[32]; snprintf(buf,sizeof(buf),"n=%d",n); r.detail=buf; }
    }
    return r;
}

static TestResult test_reverse() {
    TestResult r{"reverse/batch_reverse", 0, 0, ""};
    for (int n = 1; n <= 8; n++) {
        uint64_t p = rng();
        std::vector<int64_t> r1(n), r2(n);
        reverse_bytes8(p,n,r1.data());
        ref_reverse8(p,n,r2.data());
        r.total++;
        if (std::memcmp(r1.data(),r2.data(),n*8)==0) r.passed++;
    }
    for (int nv : {0,1,3,4,5,9,100}) {
        for (int ns = 1; ns <= 8; ns++) {
            std::vector<uint64_t> pv(nv);
            for (auto& x : pv) x = rng();
            std::vector<int64_t> r1((size_t)nv*ns), r2((size_t)nv*ns);
            batch_reverse_u8(pv.data(),nv,ns,r1.data());
            ref_batch_reverse(pv.data(),nv,ns,r2.data());
            r.total++;
            if (std::memcmp(r1.data(),r2.data(),(size_t)nv*ns*8)==0) r.passed++;
        }
    }
    return r;
}

static TestResult test_accum() {
    TestResult r{"accum_f32", 0, 0, ""};
    for (int64_t n : {0,1,7,8,9,15,16,17,100,1000}) {
        std::vector<float> d1(n,1.0f), d2(n,1.0f), s(n);
        for (auto& x : s) x = (float)(rng()%100);
        accum_f32(d1.data(),s.data(),n);
        for (int64_t i=0;i<n;i++) d2[i]+=s[i];
        r.total++;
        if (std::memcmp(d1.data(),d2.data(),n*4)==0) r.passed++;
    }
    return r;
}

// ============================================================================
// 性能基准
// ============================================================================

struct BenchResult {
    std::string name;
    size_t K;
    double us_per_call;
    double mops_per_s;
    int iters;
};

static BenchResult bench_dot16(size_t K, int iters) {
    std::vector<int16_t> a(K), b(K);
    for (size_t i = 0; i < K; i++) { a[i]=rand_range<int16_t>(-20000,20000); b[i]=rand_range<int16_t>(-20000,20000); }
    volatile int64_t sink = 0;
    for (int i = 0; i < 50; i++) sink += dot16(a.data(),b.data(),K);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) sink += dot16(a.data(),b.data(),K);
    double t1 = now_ms();
    return {"dot16", K, (t1-t0)/iters*1000, (double)K*iters/(t1-t0)/1000, iters};
}

static BenchResult bench_dot8(size_t K, int iters) {
    std::vector<uint8_t> a(K);
    std::vector<int8_t> b(K);
    for (size_t i = 0; i < K; i++) { a[i]=(uint8_t)rng(); b[i]=(int8_t)rng(); }
    volatile int64_t sink = 0;
    for (int i = 0; i < 50; i++) sink += dot8(a.data(),b.data(),K);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) sink += dot8(a.data(),b.data(),K);
    double t1 = now_ms();
    return {"dot8", K, (t1-t0)/iters*1000, (double)K*iters/(t1-t0)/1000, iters};
}

static BenchResult bench_dot4(size_t K, int iters) {
    std::vector<uint8_t> a(K);
    std::vector<int8_t> b(K);
    for (size_t i = 0; i < K; i++) { a[i]=(uint8_t)rng(); b[i]=(int8_t)rng(); }
    volatile int64_t sink = 0;
    for (int i = 0; i < 50; i++) sink += dot4(a.data(),b.data(),K);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) sink += dot4(a.data(),b.data(),K);
    double t1 = now_ms();
    return {"dot4", K, (t1-t0)/iters*1000, (double)K*iters/(t1-t0)/1000, iters};
}

static BenchResult bench_sum_f32(size_t K, int iters) {
    std::vector<float> p(K);
    for (auto& x : p) x = (float)(rng()%1000);
    volatile float sink = 0;
    for (int i = 0; i < 50; i++) sink += sum_f32(p.data(),K,1);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) sink += sum_f32(p.data(),K,1);
    double t1 = now_ms();
    return {"sum_f32", K, (t1-t0)/iters*1000, (double)K*iters/(t1-t0)/1000, iters};
}

static BenchResult bench_decode(size_t K, int iters) {
    std::vector<uint64_t> pv(K);
    std::vector<float> out(K);
    for (auto& x : pv) x = rng() & 0xFFFF;
    for (int i = 0; i < 50; i++) decode_i16_f32(pv.data(),K,0.25f,out.data());
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) decode_i16_f32(pv.data(),K,0.25f,out.data());
    double t1 = now_ms();
    return {"decode_i16_f32", K, (t1-t0)/iters*1000, (double)K*iters/(t1-t0)/1000, iters};
}

// ============================================================================
// JSON 输出
// ============================================================================

static std::string escape_json(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c=='"') r+="\\\"";
        else if (c=='\\') r+="\\\\";
        else if (c=='\n') r+="\\n";
        else r+=c;
    }
    return r;
}

int main(int argc, char** argv) {
    bool json_only = (argc > 1 && std::string(argv[1]) == "--json-only");

    // 环境信息
    const char* backend = active_backend_name();

    if (!json_only) {
        printf("=== SGN SIMD Benchmark ===\n");
        printf("后端: %s\n", backend);
        printf("============================\n\n");
    }

    // 正确性测试
    std::vector<TestResult> tests;
    tests.push_back(test_dot16());
    tests.push_back(test_dot8_dot4());
    tests.push_back(test_sum_series());
    tests.push_back(test_decode());
    tests.push_back(test_reverse());
    tests.push_back(test_accum());

    int total_tests = 0, passed_tests = 0;
    if (!json_only) printf("--- 正确性测试 ---\n");
    for (auto& t : tests) {
        total_tests += t.total;
        passed_tests += t.passed;
        if (!json_only) {
            printf("  %-20s %d/%d 通过", t.name.c_str(), t.passed, t.total);
            if (!t.detail.empty()) printf("  [失败: %s]", t.detail.c_str());
            printf("\n");
        }
    }
    if (!json_only) printf("  合计: %d/%d 通过\n\n", passed_tests, total_tests);

    // 性能基准
    std::vector<BenchResult> benches;
    size_t bench_K[] = {1024, 4096, 16384, 65536};
    for (size_t K : bench_K) {
        benches.push_back(bench_dot16(K, K > 16384 ? 2000 : 5000));
        benches.push_back(bench_dot8(K, K > 16384 ? 3000 : 8000));
        benches.push_back(bench_dot4(K, K > 16384 ? 3000 : 8000));
        benches.push_back(bench_sum_f32(K, K > 16384 ? 3000 : 8000));
        benches.push_back(bench_decode(K, K > 16384 ? 3000 : 8000));
    }

    if (!json_only) {
        printf("--- 性能基准 ---\n");
        printf("  %-18s %8s %12s %14s\n", "原语", "K", "us/call", "Mops/s");
        printf("  %s\n", std::string(56, '-').c_str());
        for (auto& b : benches) {
            printf("  %-18s %8zu %12.3f %14.1f\n", b.name.c_str(), b.K, b.us_per_call, b.mops_per_s);
        }
        printf("\n");
    }

    // 生成 JSON
    std::ostringstream json;
    json << "{\n";
    json << "  \"test_suite\": \"SGN SIMD Primitives Benchmark\",\n";
    json << "  \"backend\": \"" << backend << "\",\n";
    json << "  \"timestamp\": \"" << __DATE__ << " " << __TIME__ << "\",\n";
    json << "  \"correctness\": {\n";
    json << "    \"total\": " << total_tests << ",\n";
    json << "    \"passed\": " << passed_tests << ",\n";
    json << "    \"all_passed\": " << (passed_tests == total_tests ? "true" : "false") << ",\n";
    json << "    \"tests\": [\n";
    for (size_t i = 0; i < tests.size(); i++) {
        json << "      {\"name\": \"" << escape_json(tests[i].name) << "\", \"total\": " << tests[i].total
             << ", \"passed\": " << tests[i].passed << ", \"detail\": \"" << escape_json(tests[i].detail) << "\"}";
        if (i+1 < tests.size()) json << ",";
        json << "\n";
    }
    json << "    ]\n";
    json << "  },\n";
    json << "  \"benchmarks\": [\n";
    for (size_t i = 0; i < benches.size(); i++) {
        json << "    {\"name\": \"" << benches[i].name << "\", \"K\": " << benches[i].K
             << ", \"us_per_call\": " << benches[i].us_per_call
             << ", \"mops_per_s\": " << benches[i].mops_per_s
             << ", \"iters\": " << benches[i].iters << "}";
        if (i+1 < benches.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";

    std::string json_str = json.str();

    // 写入文件
    std::ofstream f("sgn_benchmark/result.json");
    if (f.is_open()) { f << json_str; f.close(); }

    if (json_only) {
        printf("%s", json_str.c_str());
    } else {
        printf("结果已写入 sgn_benchmark/result.json\n");
    }

    return (passed_tests == total_tests) ? 0 : 1;
}
