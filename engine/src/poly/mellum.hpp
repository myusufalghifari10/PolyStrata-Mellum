// src/poly/mellum.hpp - JetBrains' Mellum 2.1 (llama.cpp architecture "mellum"): 12B-A2.5B-Thinking-MTP.
//
// One residual stream with GQA attention in every layer, per-head RMSNorm on Q and K, and NeoX RoPE (plain on
// sliding-window layers, YaRN on full-attention layers).  Every trunk layer and the optional MTP draft block
// (nextn_predict_layers = 1) is sparse MoE with softmax top-k routing and SwiGLU, with no shared expert and no
// attention sinks.
#pragma once

#include "engine.hpp"

#include <algorithm>

namespace strata::poly {

class Mellum final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return std::max(n_head_ * head_dim_, 2 * n_embd_); }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void block_attend(int64_t pos, int n) override;
    void block_rows(int64_t pos, int n) override;
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }

private:
    struct Block {
        bool swa = false;     ///< sliding-window attention layer; its caches are rings
        bool yarn = false;    ///< full-attention layer with YaRN scaling
        Dev wq, wk, wv, wo, q_norm, k_norm;
        uint16_t* keys = nullptr;
        uint16_t* vals = nullptr;
        int8_t* k_codes = nullptr;
        uint16_t* k_scales = nullptr;
        int8_t* v_codes = nullptr;
        uint16_t* v_scales = nullptr;
    };

    void rope_heads(const Block& B, const float* x, const Dev& norm, float* y, int heads, int64_t pos, int rows,
                    const long long* at);
    void attn(Block& B, int64_t pos, int n);
    void attn_store(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void store(const float* x, const Block& B, int64_t pos, int n, bool value);
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;
    std::vector<uint8_t> swa_;   ///< per-layer SWA flag for trunk + optional MTP block
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rot_ = 0;
    float theta_full_ = 0, theta_swa_ = 0;
    RopeYarn yarn_{};
    bool yarn_on_ = false;
    bool w_norm_ = true;
    float w_scale_ = 0;
    int window_ = 0;
    int ring_ = 0;
    int keep_ = 0;
    int chunk_ = 1;

    // device scratch
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr;
    size_t score_n_ = 0;
    float* awork_ = nullptr;
    float* q8w_ = nullptr;
    size_t work_halves_ = 0;
};

}  // namespace strata::poly
