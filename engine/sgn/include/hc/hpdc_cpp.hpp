/**
 * @file hpdc_cpp.hpp
 * @brief HPDC 官方 C++ RAII 包装器（元数据驱动，零硬编码）
 * @version 1.0.0
 *
 * 设计原则：
 *   1. 所有 HC 变体（HC8/16/32/64）共用一套模板代码，层数/位数自动推导
 *   2. 运算函数通过 hc_api_traits 特化映射到 C API，新增类型只需补特化
 *   3. 与 C 层完全解耦：C 结构体加字段不影响 layers 计算（只看 v[] 数组）
 *   4. 嵌入式友好：不碰此文 = 不引入 C++ 开销
 *
 * 用法：
 *   #include "sgn/hpdc_cpp.hpp"
 *   using namespace hpdc;
 *
 *   HC8 a(3.14);          // 自动调用 hc8_from_double
 *   HC16 b(1000.0);       // 自动调用 hc16_from_double
 *   auto c = a + b;       // 编译错误：类型不匹配（设计如此）
 *   auto d = a + HC8(1.0); // OK
 *
 *   Sandbox sb;
 *   auto q = sb.divide(a, HC8(3.0));
 *   auto arr = sb.project_batch(std::vector<HC8>{...});
 */

#pragma once

#include "sgn/hpdc_core.h"
#include "sgn/hc32.h"
#include "sgn/hc64.h"
#include "sgn/hpdc_sandbox.h"
#include "sgn/hpdc_trie.h"

#ifdef SGN_USE_SIMD
#include "sgn/hc_simd.h"
#endif

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <type_traits>

namespace hpdc {

/* ============================================================================
 * 内部细节
 * ============================================================================ */
namespace detail {

/**
 * 结构体元数据萃取：从 C struct 自动推导 layers/elem_bits，零硬编码。
 * 即使 hc8_t 未来加了 level 字段，sizeof(v)/sizeof(v[0]) 仍然为 6。
 */
template<typename RawT> struct hc_struct_traits;

template<> struct hc_struct_traits<hc8_t> {
    using raw_type  = hc8_t;
    using elem_type = uint8_t;
    static constexpr size_t layers      = sizeof(raw_type::v) / sizeof(elem_type); // 6
    static constexpr size_t elem_bits   = sizeof(elem_type) * 8;                    // 8
    static constexpr size_t total_bytes = sizeof(raw_type);                         // 6
};

template<> struct hc_struct_traits<hc16_t> {
    using raw_type  = hc16_t;
    using elem_type = uint16_t;
    static constexpr size_t layers      = sizeof(raw_type::v) / sizeof(elem_type); // 4
    static constexpr size_t elem_bits   = sizeof(elem_type) * 8;                    // 16
    static constexpr size_t total_bytes = sizeof(raw_type);                         // 8
};

template<> struct hc_struct_traits<hc32_t> {
    using raw_type  = hc32_t;
    using elem_type = uint32_t;
    static constexpr size_t layers      = sizeof(raw_type::v) / sizeof(elem_type); // 3
    static constexpr size_t elem_bits   = sizeof(elem_type) * 8;                    // 32
    static constexpr size_t total_bytes = sizeof(raw_type);                         // 12
};

template<> struct hc_struct_traits<hc64_t> {
    using raw_type  = hc64_t;
    using elem_type = uint64_t;
    static constexpr size_t layers      = sizeof(raw_type::v) / sizeof(elem_type); // 2
    static constexpr size_t elem_bits   = sizeof(elem_type) * 8;                    // 64
    static constexpr size_t total_bytes = sizeof(raw_type);                         // 16
};

/* ============================================================================
 * API 映射：主模板提供通用转换，特化提供完整运算。
 * 新增 HC 类型时，只需特化此结构体即可启用全部功能。
 * ============================================================================ */
template<typename RawT>
struct hc_api_traits {
    using st = hc_struct_traits<RawT>;
    using elem_type = typename st::elem_type;

    /* 通用物理值计算：自动推导层数和进制，适用于 C 层未提供专用 to_double 的新类型 */
    static double to_double(const RawT& h) {
        double v = 0.0, scale = 1.0;
        for (size_t i = 0; i < st::layers; ++i) {
            v += static_cast<double>(h.v[i]) * scale;
            scale /= static_cast<double>(1ULL << st::elem_bits);
        }
        return v;
    }

    /* 通用饱和转换：自动推导层数 */
    static RawT from_double(double v) {
        RawT h;
        memset(&h, 0, sizeof(h));
        if (v <= 0.0) return h;
        double scaled = v;
        for (size_t i = 0; i < st::layers; ++i) {
            auto max_val = static_cast<elem_type>(-1);
            auto elem = static_cast<elem_type>(scaled);
            if (elem > max_val) elem = max_val;
            h.v[i] = elem;
            scaled = (scaled - static_cast<double>(elem)) * static_cast<double>(1ULL << st::elem_bits);
        }
        return h;
    }

    static RawT zero() {
        RawT h;
        memset(&h, 0, sizeof(h));
        return h;
    }

    static RawT max_val() {
        RawT h;
        memset(&h, 0xFF, sizeof(h));
        return h;
    }

    /* 以下运算默认不提供，必须由特化实现 */
    // static RawT add_sat(const RawT*, const RawT*);
    // static RawT sub(const RawT*, const RawT*);
    // static bool less(const RawT*, const RawT*);
    // static bool equal(const RawT*, const RawT*);
};

/* HC8：完整映射到 C API */
template<> struct hc_api_traits<hc8_t> {
    static double to_double(const hc8_t& h) { return hc8_to_double(h); }
    static hc8_t from_double(double v) { return hc8_from_double(v, SGN_OVERFLOW_SATURATE); }
    static hc8_t zero()    { return SGN_HC8_ZERO; }
    static hc8_t max_val() { return SGN_HC8_MAX; }

    static hc8_t add_sat(const hc8_t* a, const hc8_t* b) { return hc8_add_sat(a, b); }
    static hc8_t sub(const hc8_t* a, const hc8_t* b)     { return hc8_sub(a, b); }
    static bool less(const hc8_t* a, const hc8_t* b)           { return hc8_less(a, b); }
    static bool equal(const hc8_t* a, const hc8_t* b)        { return hc8_equal(a, b); }
    static hc8_t soft_threshold(const hc8_t* X, const hc8_t* L) { return hc8_soft_threshold(X, L); }
};

/* HC16：全部映射到 C API */
template<> struct hc_api_traits<hc16_t> {
    static double to_double(const hc16_t& h) { return hc16_to_double(h); }
    static hc16_t from_double(double v) { return hc16_from_double(v, SGN_OVERFLOW_SATURATE); }
    static hc16_t zero()    { return SGN_HC16_ZERO; }
    static hc16_t max_val() { return SGN_HC16_MAX; }

    static hc16_t add_sat(const hc16_t* a, const hc16_t* b) { return hc16_add_sat(a, b); }
    static hc16_t sub(const hc16_t* a, const hc16_t* b)     { return hc16_sub(a, b); }
    static bool less(const hc16_t* a, const hc16_t* b)           { return hc16_less(a, b); }
    static bool equal(const hc16_t* a, const hc16_t* b)        { return hc16_equal(a, b); }
    static hc16_t soft_threshold(const hc16_t* X, const hc16_t* L) { return hc16_soft_threshold(X, L); }
};

/* HC32：全部映射到 C API（阶段2 已补齐） */
template<> struct hc_api_traits<hc32_t> {
    static double to_double(const hc32_t& h) { return hc32_to_double(h); }
    static hc32_t from_double(double v) { return hc32_from_double(v, SGN_OVERFLOW_SATURATE); }
    static hc32_t zero()    { return SGN_HC32_ZERO; }
    static hc32_t max_val() { return SGN_HC32_MAX; }

    static hc32_t add_sat(const hc32_t* a, const hc32_t* b) { return hc32_add_sat(a, b); }
    static hc32_t sub(const hc32_t* a, const hc32_t* b)     { return hc32_sub(a, b); }
    static bool less(const hc32_t* a, const hc32_t* b)           { return hc32_less(a, b); }
    static bool equal(const hc32_t* a, const hc32_t* b)        { return hc32_equal(a, b); }
    static hc32_t soft_threshold(const hc32_t* X, const hc32_t* L) { return hc32_soft_threshold(X, L); }
};

/* HC64：全部映射到 C API */
template<> struct hc_api_traits<hc64_t> {
    static double to_double(const hc64_t& h) { return hc64_to_double(h); }
    static hc64_t from_double(double v) { return hc64_from_double(v, SGN_OVERFLOW_SATURATE); }
    static hc64_t zero()    { return SGN_HC64_ZERO; }
    static hc64_t max_val() { return SGN_HC64_MAX; }

    static hc64_t add_sat(const hc64_t* a, const hc64_t* b) { return hc64_add_sat(a, b); }
    static hc64_t sub(const hc64_t* a, const hc64_t* b)     { return hc64_sub(a, b); }
    static bool less(const hc64_t* a, const hc64_t* b)           { return hc64_less(a, b); }
    static bool equal(const hc64_t* a, const hc64_t* b)        { return hc64_equal(a, b); }
    static hc64_t soft_threshold(const hc64_t* X, const hc64_t* L) { return hc64_soft_threshold(X, L); }
};

/* ============================================================================
 * Sandbox 内部重载分派（避免模板技巧，直接函数重载）
 * ============================================================================ */
inline double sb_project(sandbox_t* sb, const hc8_t& h)  { return sandbox_project_hc8(sb, h); }
inline double sb_project(sandbox_t* sb, const hc16_t& h) { return sandbox_project_hc16(sb, h); }
inline double sb_project(sandbox_t* sb, const hc32_t& h) { return sandbox_project_hc32(sb, h); }
inline double sb_project(sandbox_t* sb, const hc64_t& h) { return hc64_to_double(h) / sb->R; }

inline hc8_t  sb_inverse(sandbox_t* sb, double phi, const hc8_t&)  { return sandbox_inverse_hc8(sb, phi); }
inline hc16_t sb_inverse(sandbox_t* sb, double phi, const hc16_t&) { return sandbox_inverse_hc16(sb, phi); }
inline hc32_t sb_inverse(sandbox_t* sb, double phi, const hc32_t&) { return sandbox_inverse_hc32(sb, phi); }
inline hc64_t sb_inverse(sandbox_t* sb, double phi, const hc64_t&) {
    if (phi < 0.0) phi = 0.0;
    if (phi >= 1.0) phi = 0.9999999999999999;
    return hc64_from_double(phi * sb->R, SGN_OVERFLOW_SATURATE);
}

inline hc8_t  sb_divide(sandbox_t* sb, const hc8_t& a, const hc8_t& b)  { return sandbox_divide_hc8(sb, a, b); }
inline hc16_t sb_divide(sandbox_t* sb, const hc16_t& a, const hc16_t& b) { return sandbox_divide_hc16(sb, a, b); }
inline hc32_t sb_divide(sandbox_t* sb, const hc32_t& a, const hc32_t& b) { return sandbox_divide_hc32(sb, a, b); }
inline hc64_t sb_divide(sandbox_t* sb, const hc64_t& a, const hc64_t& b) { return sandbox_divide_hc64(sb, a, b); }

inline hc8_t  sb_gradient(sandbox_t* sb, const hc8_t& h, double g, double e)  { return sandbox_gradient_hc8(sb, h, g, e); }
inline hc16_t sb_gradient(sandbox_t* sb, const hc16_t& h, double g, double e) { return sandbox_gradient_hc16(sb, h, g, e); }
inline hc32_t sb_gradient(sandbox_t* sb, const hc32_t& h, double g, double e) { return sandbox_gradient_hc32(sb, h, g, e); }
inline hc64_t sb_gradient(sandbox_t* sb, const hc64_t& h, double g, double e) { return sandbox_gradient_hc64(sb, h, g, e); }

inline hc8_t  sb_scale(sandbox_t* sb, const hc8_t& h, double f)  { return sandbox_scale_hc8(sb, h, f); }
inline hc16_t sb_scale(sandbox_t* sb, const hc16_t& h, double f) { return sandbox_scale_hc16(sb, h, f); }
inline hc32_t sb_scale(sandbox_t* sb, const hc32_t& h, double f) { return sandbox_scale_hc32(sb, h, f); }
inline hc64_t sb_scale(sandbox_t* sb, const hc64_t& h, double f) { return sandbox_scale_hc64(sb, h, f); }

} /* namespace detail */

/* ============================================================================
 * HC<T> 通用包装器
 * ============================================================================ */
template<typename RawT>
class HC {
    using st  = detail::hc_struct_traits<RawT>;
    using api = detail::hc_api_traits<RawT>;

    RawT raw_;

public:
    using raw_type  = RawT;
    using elem_type = typename st::elem_type;
    static constexpr size_t layers    = st::layers;
    static constexpr size_t elem_bits   = st::elem_bits;
    static constexpr size_t total_bytes = st::total_bytes;

    HC() = default;
    explicit HC(double v) : raw_(api::from_double(v)) {}
    explicit HC(const RawT& r) : raw_(r) {}

    /* 转换 */
    double to_float() const { return api::to_double(raw_); }

    /* 层访问：边界检查自动使用推导的 layers */
    elem_type operator[](size_t i) const {
        if (i >= layers) throw std::out_of_range("HC layer index out of range");
        return raw_.v[i];
    }
    void set_layer(size_t i, elem_type v) {
        if (i >= layers) throw std::out_of_range("HC layer index out of range");
        raw_.v[i] = v;
    }

    /* 比较（仅当 api 提供 less/equal 时可用） */
    bool operator<(const HC& o) const  { return api::less(&raw_, &o.raw_); }
    bool operator==(const HC& o) const { return api::equal(&raw_, &o.raw_); }
    bool operator!=(const HC& o) const { return !(*this == o); }
    bool operator<=(const HC& o) const { return !(o < *this); }
    bool operator>(const HC& o) const  { return o < *this; }
    bool operator>=(const HC& o) const { return !(*this < o); }

    /* 算术（仅当 api 提供 add_sat/sub 时可用） */
    HC operator+(const HC& o) const { return HC(api::add_sat(&raw_, &o.raw_)); }
    HC operator-(const HC& o) const { return HC(api::sub(&raw_, &o.raw_)); }
    HC& operator+=(const HC& o) { raw_ = api::add_sat(&raw_, &o.raw_); return *this; }
    HC& operator-=(const HC& o) { raw_ = api::sub(&raw_, &o.raw_); return *this; }

    /* 序列化 */
    std::string to_bytes() const {
        return std::string(reinterpret_cast<const char*>(&raw_), sizeof(raw_));
    }
    static HC from_bytes(const std::string& s) {
        HC h;
        if (s.size() >= sizeof(h.raw_)) {
            memcpy(&h.raw_, s.data(), sizeof(h.raw_));
        }
        return h;
    }

    /* 工厂 */
    static HC zero()    { return HC(api::zero()); }
    static HC max_val() { return HC(api::max_val()); }

    /* 原始 C 类型访问（用于和 C API 互操作） */
    const RawT& raw() const { return raw_; }
    RawT& raw() { return raw_; }

    /* 字符串表示 */
    std::string repr() const {
        std::ostringstream ss;
        ss << "HC" << elem_bits << "(" << to_float() << ")";
        return ss.str();
    }
};

/* ============================================================================
 * DC 包装器
 * ============================================================================ */
class DC {
    dc_t raw_;

public:
    DC() = default;
    DC(int64_t idx, uint32_t lvl) : raw_{idx, lvl} {}
    explicit DC(double v, uint32_t lvl) : raw_(dc_from_double(v, lvl)) {}
    explicit DC(const dc_t& r) : raw_(r) {}

    double to_float() const { return dc_to_double(&raw_); }
    int64_t index() const { return raw_.index; }
    uint32_t level() const { return raw_.level; }

    std::string to_json() const {
        char buf[64];
        dc_serialize(&raw_, buf, sizeof(buf));
        return std::string(buf);
    }

    DC to_level(uint32_t target) const { return DC(dc_to_level(&raw_, target)); }

    bool operator<(const DC& o) const  { return dc_less(&raw_, &o.raw_); }
    bool operator==(const DC& o) const { return dc_equal(&raw_, &o.raw_); }
    bool operator!=(const DC& o) const { return !(*this == o); }

    DC operator+(const DC& o) const { return DC(dc_add(&raw_, &o.raw_)); }
    DC operator-(const DC& o) const { return DC(dc_sub(&raw_, &o.raw_)); }
    DC operator*(const DC& o) const { return DC(dc_mul(&raw_, &o.raw_)); }

    const dc_t& raw() const { return raw_; }
    dc_t& raw() { return raw_; }
};

/* ============================================================================
 * Sandbox RAII 包装器
 * ============================================================================ */
class Sandbox {
    sandbox_t* sb_;

public:
    Sandbox(sgn_precision_t prec = SGN_PREC_STD, uint32_t int_bits = 8)
        : sb_(sandbox_create(prec, int_bits)) {
        if (!sb_) throw std::runtime_error("Failed to create sandbox");
    }
    ~Sandbox() { sandbox_destroy(sb_); }

    Sandbox(const Sandbox&) = delete;
    Sandbox& operator=(const Sandbox&) = delete;
    Sandbox(Sandbox&& o) noexcept : sb_(o.sb_) { o.sb_ = nullptr; }
    Sandbox& operator=(Sandbox&& o) noexcept {
        if (this != &o) { sgn_sandbox_destroy(sb_); sb_ = o.sb_; o.sb_ = nullptr; }
        return *this;
    }

    /* 投影 / 逆投影（模板自动分派） */
    template<typename HCType>
    double project(const HCType& h) const {
        return detail::sb_project(sb_, h.raw());
    }

    template<typename HCType>
    HCType inverse(double phi) const {
        return HCType(detail::sb_inverse(sb_, phi, typename HCType::raw_type{}));
    }

    /* 算术 */
    template<typename HCType>
    HCType divide(const HCType& a, const HCType& b) const {
        return HCType(detail::sb_divide(sb_, a.raw(), b.raw()));
    }

    template<typename HCType>
    HCType gradient(const HCType& h, double grad, double eta) const {
        return HCType(detail::sb_gradient(sb_, h.raw(), grad, eta));
    }

    template<typename HCType>
    HCType scale(const HCType& h, double factor) const {
        return HCType(detail::sb_scale(sb_, h.raw(), factor));
    }

    /* 有符号运算 */
    shc8_t signed_add(const shc8_t& a, const shc8_t& b) const {
        return sgn_sandbox_signed_add(sb_, a, b);
    }
    shc8_t signed_sub(const shc8_t& a, const shc8_t& b) const {
        return sgn_sandbox_signed_sub(sb_, a, b);
    }

    /* 泛型 float 映射 */
    template<typename HCType, typename Fn>
    HCType map(const HCType& h, Fn fn) const {
        return inverse<HCType>(fn(project(h)));
    }

    template<typename HCType, typename Fn>
    HCType map2(const HCType& a, const HCType& b, Fn fn) const {
        return inverse<HCType>(fn(project(a), project(b)));
    }

    /* 批量操作 */
    template<typename HCType>
    std::vector<double> project_batch(const std::vector<HCType>& in) const {
        std::vector<double> out(in.size());
        for (size_t i = 0; i < in.size(); ++i) out[i] = project(in[i]);
        return out;
    }

    template<typename HCType>
    std::vector<HCType> inverse_batch(const std::vector<double>& in) const {
        std::vector<HCType> out;
        out.reserve(in.size());
        for (double v : in) out.push_back(inverse<HCType>(v));
        return out;
    }

    /* DC 桥梁 */
    DC hc_to_dc(const HC<hc8_t>& h, uint32_t level) const {
        return DC(hc8_to_dc(&h.raw(), level));
    }
    HC<hc8_t> dc_to_hc(const DC& d, sgn_precision_t prec = SGN_PREC_STD) const {
        return HC<hc8_t>(dc_to_hc8(&d.raw(), prec));
    }

    /* 方案切换 */
    int set_scheme(const std::string& name) {
        return sandbox_set_scheme(sb_, name.c_str());
    }
    std::string get_scheme() const {
        const char* s = sandbox_get_scheme(sb_);
        return s ? std::string(s) : "default";
    }
};

/* ============================================================================
 * TriePool 包装器
 * ============================================================================ */
class TriePool {
    trie_pool_t pool_;

public:
    TriePool() { trie_pool_init(&pool_); }
    TriePool(const TriePool&) = delete;
    TriePool& operator=(const TriePool&) = delete;
    /* 不实现移动：内部指针自引用，移动后悬挂 */

    bool insert(const HC<hc8_t>& sig, uint16_t tid) {
        return trie_insert_template(&pool_.nodes[0], &sig.raw(), tid, &pool_);
    }

    trie_pool_t* native() { return &pool_; }
    const trie_pool_t* native() const { return &pool_; }
};

/* ============================================================================
 * Plugin RAII 包装器
 * ============================================================================ */
class Plugin {
    plugin_handle_t* h_;

public:
    explicit Plugin(const std::string& path)
        : h_(plugin_load(path.c_str())) {
        if (!h_) throw std::runtime_error("Failed to load plugin: " + path);
    }
    ~Plugin() {
        if (h_) plugin_unload(h_);
    }

    Plugin(const Plugin&) = delete;
    Plugin& operator=(const Plugin&) = delete;
    Plugin(Plugin&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Plugin& operator=(Plugin&& o) noexcept {
        if (this != &o) {
            if (h_) plugin_unload(h_);
            h_ = o.h_;
            o.h_ = nullptr;
        }
        return *this;
    }

    plugin_handle_t* native() const { return h_; }
    bool is_valid() const { return h_ != nullptr; }
};

/* ============================================================================
 * 类型别名（向后兼容）
 * ============================================================================ */
using HC8  = HC<hc8_t>;
using HC16 = HC<hc16_t>;
using HC32 = HC<hc32_t>;
using HC64 = HC<hc64_t>;

/* ============================================================================
 * 便利函数
 * ============================================================================ */
template<typename RawT>
HC<RawT> soft_threshold(const HC<RawT>& X, const HC<RawT>& Lambda) {
    return HC<RawT>(detail::hc_api_traits<RawT>::soft_threshold(&X.raw(), &Lambda.raw()));
}

/* ============================================================================
 * SIMD 批量操作（需定义 SGN_USE_SIMD 并链接 hc_simd.c）
 * ============================================================================ */
#ifdef SGN_USE_SIMD

/** 批量 HC16 饱和加法 */
inline std::vector<HC16> add_batch(const std::vector<HC16>& a, const std::vector<HC16>& b) {
    size_t n = std::min(a.size(), b.size());
    std::vector<HC16> out(n);
    hc16_add_sat_batch(&a[0].raw(), &b[0].raw(), &out[0].raw(), (uint32_t)n);
    return out;
}

/** 批量 HC16 软阈值 */
inline std::vector<HC16> soft_threshold_batch(const std::vector<HC16>& a,
                                               const std::vector<HC16>& Lambda) {
    size_t n = std::min(a.size(), Lambda.size());
    std::vector<HC16> out(n);
    hc16_soft_threshold_batch(&a[0].raw(), &Lambda[0].raw(), &out[0].raw(), (uint32_t)n);
    return out;
}

/** 批量 HC16 标量缩放 */
inline std::vector<HC16> scale_batch(const std::vector<HC16>& a, double factor) {
    size_t n = a.size();
    std::vector<HC16> out(n);
    uint32_t factor_q16 = (uint32_t)(factor * 65536.0);
    hc16_scale_batch(&a[0].raw(), factor_q16, &out[0].raw(), (uint32_t)n);
    return out;
}

#endif /* SGN_USE_SIMD */

/* ============================================================================
 * short 命名空间：极简便利层（类似 Sandbox 的 map/map2 体验）
 *
 * 用法：
 *   using namespace hpdc::op;
 *   HC8 a(3.14), b(2.72);
 *   auto c = add(a, b);        // 7 字符，比 hc8_add_sat(&a,&b) 短 60%
 *   bool ok = less(a, b);
 *   HC8 st = thresh(a, b);     // 软阈值
 * ============================================================================ */
namespace op {

template<typename T> T add(const T& a, const T& b)  { return a + b; }
template<typename T> T sub(const T& a, const T& b)  { return a - b; }
template<typename T> bool less(const T& a, const T& b) { return a < b; }
template<typename T> bool eq(const T& a, const T& b)   { return a == b; }
template<typename T> T from(double v) { return T(v); }
template<typename T> double to(const T& v) { return v.to_float(); }
template<typename T> T thresh(const T& x, const T& l) { return soft_threshold(x, l); }

} /* namespace op */

} /* namespace hpdc */
