// src/poly/mellum.cpp - the layers of Mellum 2.1.  The reference for the math is llama.cpp's mellum graph
// (src/models/mellum.cpp) and its RoPE / YaRN scaling (laguna.cpp / ggml rope_yarn).
#include "mellum.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
constexpr size_t kQ8WorkRows = 2048;               // the K or V blocks a q8 prompt batch dequantizes at a time
constexpr int64_t kGatingSoftmax = 1;              // llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX
constexpr int kPeriod = 4;                         // default SWA 3:1 pattern: full at il % 4 == 3
constexpr float kPi = 3.14159265358979323846f;

int64_t of_layer_opt(const MetaValue& v, int il, int64_t fallback) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) return fallback;
    return (int64_t) v.items[(size_t) il].u;
}

float yarn_corr_dim(int n_dims, float n_ctx_orig, float n_rot, float base) {
    return (float) n_dims * std::log(n_ctx_orig / (n_rot * 2.0f * kPi)) / (2.0f * std::log(base));
}

}  // namespace

void Mellum::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

void Mellum::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto num = [&](const char* key, double otherwise) {
        const MetaValue* v = find(key);
        return v && v->is_num() ? v->num() : otherwise;
    };

    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(n_head_ > 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention.head_count(_kv)");
    need(shape_i("attention.value_length") == head_dim_, "the keys and values have different widths");
    need(head_dim_ % 32 == 0 && head_dim_ <= 256, "bad head size");

    const MetaValue* rot = find("rope.dimension_count");
    n_rot_ = rot ? (int) rot->u : head_dim_;
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && n_rot_ <= head_dim_, "bad rope.dimension_count");

    const MetaValue* window = find("attention.sliding_window");
    window_ = window ? (int) window->u : 0;
    need(window_ >= 0, "bad attention.sliding_window");

    const MetaValue* nextn = find("nextn_predict_layers");
    const int n_meta = n_layer_ + (nextn ? (int) nextn->u : 0);
    const int n_all = n_layer_ + (block_layer_ ? 1 : 0);
    swa_.assign((size_t) n_all, 0);
    if (window_ > 0) {
        const MetaValue* pattern = find("attention.sliding_window_pattern");
        if (pattern && pattern->type == MetaType::ARRAY) {
            need((int) pattern->items.size() == n_layer_ || (int) pattern->items.size() == n_meta,
                 "bad attention.sliding_window_pattern length");
        }
        for (int il = 0; il < n_all; ++il) {
            if (pattern) {
                // If pattern omits the MTP layer, MTP defaults to sliding_attention (conversion/mellum.py:80).
                swa_[(size_t) il] = of_layer_opt(*pattern, il, 1) != 0 ? 1 : 0;
            } else {
                // Default 3:1 pattern for trunk (full at il % 4 == 3), sliding for MTP.
                swa_[(size_t) il] = (il < n_layer_ ? (il % kPeriod != kPeriod - 1) : true) ? 1 : 0;
            }
        }
    }

    const float base_full = shape_f("rope.freq_base");
    const float base_swa = (float) num("rope.freq_base_swa", base_full);
    theta_full_ = std::pow(base_full, -2.f / (float) n_rot_);
    theta_swa_ = std::pow(base_swa, -2.f / (float) n_rot_);

    const MetaValue* scaling = find("rope.scaling.type");
    const std::string kind = scaling && scaling->type == MetaType::STRING ? scaling->s : std::string("none");
    need(kind == "none" || kind == "yarn" || kind == "linear", "rope.scaling.type is not none, linear or yarn");
    const float factor = (float) num("rope.scaling.factor", 1.0);
    need(kind == "none" || factor > 0.f, "bad rope.scaling.factor");
    yarn_ = RopeYarn{};
    yarn_on_ = false;
    if (kind != "none" && factor != 1.f) {
        yarn_on_ = true;
        yarn_.freq_scale = 1.0f / factor;
        float attn = (float) num("rope.scaling.yarn_attn_factor", 1.0);
        if (kind == "yarn") {
            yarn_.ext_factor = 1.0f;
            auto mscale = [](float scale, float m) { return scale <= 1.0f ? 1.0f : 0.1f * m * std::log(scale) + 1.0f; };
            const float log_mul = (float) num("rope.scaling.yarn_log_multiplier", 0.0);
            attn = log_mul != 0.f ? mscale(factor, 1.0f) / mscale(factor, log_mul) : mscale(factor, 1.0f);
            attn *= 1.0f / (1.0f + 0.1f * std::log(factor));
            const MetaValue* orig = find("rope.scaling.original_context_length");
            const float n_orig = (float) (orig ? orig->u : (uint64_t) shape_i("context_length"));
            const float fast = (float) num("rope.scaling.yarn_beta_fast", 32.0);
            const float slow = (float) num("rope.scaling.yarn_beta_slow", 1.0);
            yarn_.corr_low = std::max(0.0f, std::floor(yarn_corr_dim(n_rot_, n_orig, fast, base_full)));
            yarn_.corr_high =
                std::min((float) (n_rot_ - 1), std::ceil(yarn_corr_dim(n_rot_, n_orig, slow, base_full)));
        }
        attn *= (float) num("rope.scaling.attn_factor", 1.0);
        yarn_.mscale = yarn_.ext_factor != 0.f ? attn * (1.0f + 0.1f * std::log(1.0f / yarn_.freq_scale)) : attn;
        std::fprintf(stderr,
                     "strata-poly: mellum: full layers turn %d of %d values by %s (factor %g, pairs %g to %g, "
                     "magnitude %.4f); window layers plain\n",
                     n_rot_, head_dim_, kind.c_str(), (double) factor, (double) yarn_.corr_low,
                     (double) yarn_.corr_high, (double) yarn_.mscale);
    }

    const MetaValue* gating = find("expert_gating_func");
    need(!gating || (int64_t) gating->u == kGatingSoftmax, "the expert gate is not softmax");
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : true;
    w_scale_ = (float) num("expert_weights_scale", 0.0);

    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    half_sum_guard_ = true;
}

void Mellum::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    B.swa = swa_[(size_t) il] != 0;
    B.yarn = !B.swa && yarn_on_;
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    L.fixed = true;
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.q_norm = f32("attn_q_norm.weight", head_dim_);
    B.k_norm = f32("attn_k_norm.weight", head_dim_);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
    load_moe(il, /*shared=*/false);
}

void Mellum::alloc_caches() {
    ring_ = window_ > 0 ? std::min(opt_.max_context, window_ + rows_) : 0;
    keep_ = std::min(window_, ring_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t row = (size_t) n_kv_ * head_dim_;
    const size_t scale_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
    if (q8) std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", row + scale_row);
    size_t rings = 0, codes = 0, scales = 0;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        if (!runs(il) || !blocks_[(size_t) il].swa) continue;
        if (q8) {
            codes += 2 * (size_t) ring_ * row;
            scales += 2 * (size_t) ring_ * scale_row;
        } else {
            rings += 2 * (size_t) ring_ * row;
        }
    }
    uint8_t* rc = codes ? (uint8_t*) seq_alloc(codes, "the attention cache") : nullptr;
    uint8_t* rs = scales ? (uint8_t*) seq_alloc(scales, "the attention cache") : nullptr;
    uint16_t* kv = rings ? (uint16_t*) seq_alloc(rings * sizeof(uint16_t), "the attention cache") : nullptr;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.swa) {
            if (q8) {
                B.k_codes = (int8_t*) rc;
                rc += (size_t) ring_ * row;
                B.k_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
                B.v_codes = (int8_t*) rc;
                rc += (size_t) ring_ * row;
                B.v_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
            } else {
                B.keys = kv;
                kv += ring_ * row;
                B.vals = kv;
                kv += ring_ * row;
            }
        } else if (q8) {
            B.k_codes = (int8_t*) kv_alloc(row, 1);
            B.k_scales = (uint16_t*) kv_alloc(scale_row, 1);
            B.v_codes = (int8_t*) kv_alloc(row, 1);
            B.v_scales = (uint16_t*) kv_alloc(scale_row, 1);
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

void Mellum::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_n_ = (size_t) chunk_ * n_head_ * opt_.max_context;
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);
    if (rows_ >= kStageRows && opt_.kv_format == KvFmt::kQ8_0) {
        work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
        q8w_ = fbuf((2 * work_halves_ + 1) / 2);
    }
}

size_t Mellum::card_row_bytes() const {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    size_t layers = 0;
    for (const Block& B : blocks_)
        if ((q8 ? B.k_codes != nullptr : B.keys != nullptr) && !B.swa) ++layers;
    return layers * 2 * (code_row + scale_row);
}

void Mellum::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    for (const Block& B : blocks_) {
        if (B.swa || n == 0 || !(q8 ? B.k_codes != nullptr : B.keys != nullptr)) continue;
        if (q8) {
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            continue;
        }
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
        }
    }
}

void Mellum::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    const size_t sk_c = (size_t) (first - from) * code_row, sk_s = (size_t) (first - from) * scale_row;
    const size_t all_c = (size_t) count * code_row, all_s = (size_t) count * scale_row;
    for (const Block& B : blocks_) {
        if (B.swa || n == 0 || !(q8 ? B.k_codes != nullptr : B.keys != nullptr)) continue;
        if (q8) {
            ck(cudaMemcpy((uint8_t*) B.k_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.k_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            ck(cudaMemcpy((uint8_t*) B.v_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.v_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            continue;
        }
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
        }
    }
}

template <class F>
void Mellum::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

size_t Mellum::extra_bytes() const {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.swa) ++layers;
    return layers * 2 * (size_t) keep_ * (code_row + scale_row);
}

void Mellum::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (!blocks_[(size_t) il].swa) return;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t whole = 2 * (size_t) keep_ * (code_row + scale_row);
    dst += state_n_ * sizeof(float);
    for (int j = 0; j < il; ++j)
        if (blocks_[(size_t) j].swa) dst += whole;
    const Block& B = blocks_[(size_t) il];
    auto save_ring = [&](const uint8_t* cache, size_t one_row) {
        std::memset(dst, 0, (size_t) keep_ * one_row);
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpyAsync(dst + slot * one_row, cache + at * one_row, m * one_row, cudaMemcpyDeviceToHost,
                               cs(st_)),
               "save state");
        });
        dst += (size_t) keep_ * one_row;
    };
    if (q8) {
        save_ring((const uint8_t*) B.k_codes, code_row);
        save_ring((const uint8_t*) B.k_scales, scale_row);
        save_ring((const uint8_t*) B.v_codes, code_row);
        save_ring((const uint8_t*) B.v_scales, scale_row);
        return;
    }
    save_ring((const uint8_t*) B.keys, code_row);
    save_ring((const uint8_t*) B.vals, code_row);
}

void Mellum::load_extra(const uint8_t* src, int64_t pos) {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        auto load_ring = [&](uint8_t* cache, size_t one_row) {
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy(cache + at * one_row, src + slot * one_row, m * one_row, cudaMemcpyHostToDevice),
                   "load state");
            });
            src += (size_t) keep_ * one_row;
        };
        if (q8) {
            load_ring((uint8_t*) B.k_codes, code_row);
            load_ring((uint8_t*) B.k_scales, scale_row);
            load_ring((uint8_t*) B.v_codes, code_row);
            load_ring((uint8_t*) B.v_scales, scale_row);
            continue;
        }
        load_ring((uint8_t*) B.keys, code_row);
        load_ring((uint8_t*) B.vals, code_row);
    }
}

int Mellum::head_key(int il, int n) const { return blocks_[(size_t) il].swa ? 0 : room_key(n); }

void Mellum::store(const float* x, const Block& B, int64_t pos, int n, bool value) {
    const int width = n_kv_ * head_dim_;
    if (opt_.kv_format == KvFmt::kQ8_0) {
        int8_t* codes = value ? B.v_codes : B.k_codes;
        uint16_t* scales = value ? B.v_scales : B.k_scales;
        if (recording_) return store_q8_at(x, codes, scales, width, at_, n, B.swa ? ring_ : 0, st_);
        if (!B.swa) return store_q8(x, codes, scales, width, pos, n, st_);
        const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
        store_q8(x, codes, scales, width, at, head, st_);
        if (head < n) store_q8(x + (size_t) head * width, codes, scales, width, 0, n - head, st_);
        return;
    }
    uint16_t* cache = value ? B.vals : B.keys;
    if (recording_) return store_f16_at(x, cache, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

void Mellum::rope_heads(const Block& B, const float* x, const Dev& norm, float* y, int heads, int64_t pos, int rows,
                        const long long* at) {
    if (B.yarn)
        return head_norm_rope_yarn(x, head_dim_, fp(norm), y, head_dim_, heads, n_rot_, theta_full_, pos, rows,
                                   rms_eps_, st_, yarn_, at);
    const float theta = B.swa ? theta_swa_ : theta_full_;
    head_norm_rope(x, head_dim_, fp(norm), y, head_dim_, heads, n_rot_, theta, pos, rows, rms_eps_, st_, nullptr,
                   nullptr, 1, at);
}

void Mellum::attn_store(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;
    lin_pair(B.wk, B.wv, xn_, kraw_, v_, n, xq_);
    rope_heads(B, kraw_, B.k_norm, k_, n_kv_, pos, n, at ? at + 1 : nullptr);
    store(k_, B, pos, n, false);
    store(v_, B, pos, n, true);
}

void Mellum::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);
    const long long* at = recording_ ? at_ : nullptr;
    const long long* rot = at ? at + 1 : nullptr;
    lin(B.wq, xn_, qraw_, n, xq_);
    rope_heads(B, qraw_, B.q_norm, q_, n_head_, pos, n, rot);
    lin_pair(B.wk, B.wv, xn_, kraw_, v_, n, xq_);
    rope_heads(B, kraw_, B.k_norm, k_, n_kv_, pos, n, rot);
    store(k_, B, pos, n, false);
    store(v_, B, pos, n, true);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const Kvq8 kq = q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{};
    const Kvq8 vq = q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{};
    if (n >= kStageRows)
        attend_batch(blas_, q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, score_n_, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_, window, ring, false, kq, vq,
                     q8 ? (uint16_t*) q8w_ : nullptr, q8 ? (uint16_t*) q8w_ + work_halves_ : nullptr,
                     q8 ? work_halves_ : 0);
    else
        attend(q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_,
               scale, st_, window, ring, false, at,
               !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n), score_n_, kq, vq);
    lin(B.wo, att_, ao_, n);
}

void Mellum::attn_rows(Block& B, int n) {
    lin(B.wq, xn_, qraw_, n, xq_);
    lin_pair(B.wk, B.wv, xn_, kraw_, v_, n, xq_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const size_t kv = (size_t) n_kv_ * head_dim_, hq = (size_t) n_head_ * head_dim_;
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at_ + 2 * i;
        rope_heads(B, qraw_ + i * hq, B.q_norm, q_ + i * hq, n_head_, 0, 1, ai + 1);
        rope_heads(B, kraw_ + i * kv, B.k_norm, k_ + i * kv, n_kv_, 0, 1, ai + 1);
        uint16_t* kch = q8 ? nullptr : (uint16_t*) seq_buf(r.seq, B.keys);
        uint16_t* vch = q8 ? nullptr : (uint16_t*) seq_buf(r.seq, B.vals);
        const Kvq8 kq = q8 ? Kvq8{(int8_t*) seq_buf(r.seq, B.k_codes), (uint16_t*) seq_buf(r.seq, B.k_scales)}
                           : Kvq8{};
        const Kvq8 vq = q8 ? Kvq8{(int8_t*) seq_buf(r.seq, B.v_codes), (uint16_t*) seq_buf(r.seq, B.v_scales)}
                           : Kvq8{};
        if (q8) {
            store_q8_at(k_ + i * kv, (int8_t*) kq.codes, (uint16_t*) kq.scales, (int) kv, ai, 1, ring, st_);
            store_q8_at(v_ + i * kv, (int8_t*) vq.codes, (uint16_t*) vq.scales, (int) kv, ai, 1, ring, st_);
        } else {
            store_f16_at(k_ + i * kv, kch, (int) kv, ai, 1, ring, st_);
            store_f16_at(v_ + i * kv, vch, (int) kv, ai, 1, ring, st_);
        }
        const int64_t keep = pos_;
        pos_ = r.pos;
        const int sized = B.swa ? std::min(window_, opt_.max_context) : room(1);
        pos_ = keep;
        attend(q_ + i * hq, kch, vch, r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_,
               window, ring, false, ai, sized, score_n_, kq, vq);
    }
    lin(B.wo, att_, ao_, n);
}

void Mellum::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);

    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

void Mellum::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Mellum::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void Mellum::route(int, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    float mx = logits[0];
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e]);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs[e] = std::exp(logits[e] - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = probs[best] * inv;
        sum += w[k];
        probs[best] = -1.f;
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool Mellum::route_card(int, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 0, 1.f, nullptr, nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

void Mellum::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

void Mellum::block_attend(int64_t pos, int n) { attn(blocks_[(size_t) n_layer_], pos, n); }
void Mellum::block_rows(int64_t pos, int n) { attn_store(blocks_[(size_t) n_layer_], pos, n); }

}  // namespace strata::poly
