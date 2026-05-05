// Reproducer for the Mali-G715 (Pixel 9 Pro) Vulkan first-submit
// `vk::Queue::submit: ErrorDeviceLost` observed by qvac-lib-infer-llamacpp-embed
// when running gte-large_fp16 in encoder/embedding mode at batch_size=1024
// with five ~272-token sequences packed into a single submit.
//
// Same compute graph runs cleanly on:
//   - CPU on the same Pixel 9 Pro
//   - Adreno 750 (Samsung Galaxy S25 Ultra) via the same code path
//   - NVIDIA / Apple Metal / desktop Vulkan
//
// Pass:  llama_decode succeeds, embeddings are non-NaN.
// Fail:  llama_decode returns < 0 (typically when the Vulkan submit returns
//        VK_ERROR_DEVICE_LOST), or any embedding contains NaN/Inf.
//
// Usage: provide the gguf path either as argv[1] or via $LLAMACPP_TEST_MODELFILE.
//        Optional --batch <N> and --seqs <N> override the defaults that
//        match the original repro (batch_size=1024, 5 sequences).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "get-model.h"
#include "llama.h"

namespace {

struct args {
    const char * model_path = nullptr;
    int batch_size = 1024;
    int n_sequences = 5;
    int n_ctx = 512;
};

args parse_args(int argc, char ** argv) {
    args a;
    a.model_path = get_model_or_exit(argc, argv);
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            a.batch_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seqs") == 0 && i + 1 < argc) {
            a.n_sequences = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            a.n_ctx = std::atoi(argv[++i]);
        }
    }
    return a;
}

// Mimic the embed addon's per-sequence text shape:
//   maxContextSize = 512, repeatCount = ceil(512/6 + 50) = 136, "Hello world "
//   repeated → 1632 chars ≈ 272 BPE tokens for gte-large.
std::vector<std::string> build_prompts(int n) {
    static const char * variants[] = {
        "Hello world ", "Hello there ", "Hello again ", "Hello other ", "Hello earth ",
        "Hello cosmos ", "Hello galaxy ", "Hello planet "
    };
    constexpr int repeat = 136;
    std::vector<std::string> prompts;
    prompts.reserve(n);
    for (int i = 0; i < n; ++i) {
        const char * v = variants[i % (sizeof(variants) / sizeof(variants[0]))];
        std::string s;
        s.reserve(std::strlen(v) * repeat);
        for (int r = 0; r < repeat; ++r) s += v;
        prompts.push_back(std::move(s));
    }
    return prompts;
}

bool any_nan_or_inf(const float * data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) return true;
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    const args a = parse_args(argc, argv);

    fprintf(stderr,
            "[mali-repro] model=%s batch_size=%d n_sequences=%d n_ctx=%d\n",
            a.model_path, a.batch_size, a.n_sequences, a.n_ctx);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999; // matches the failing addon config
    llama_model * model = llama_model_load_from_file(a.model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "[mali-repro] failed to load model\n");
        llama_backend_free();
        return EXIT_FAILURE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = a.n_ctx;
    cparams.n_batch   = a.batch_size;
    cparams.n_ubatch  = a.batch_size; // bert-encoder requires ubatch == batch
    cparams.n_seq_max = a.n_sequences;
    cparams.embeddings    = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // matches addon config

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "[mali-repro] failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    // Tokenize the synthetic prompts.
    std::vector<std::vector<llama_token>> inputs;
    inputs.reserve(a.n_sequences);
    const std::vector<std::string> prompts = build_prompts(a.n_sequences);
    for (size_t i = 0; i < prompts.size(); ++i) {
        std::vector<llama_token> inp = common_tokenize(ctx, prompts[i], true, true);
        if ((int) inp.size() > a.n_ctx) {
            // Trim to ctx; the addon-side test does the same implicitly.
            inp.resize(a.n_ctx);
        }
        fprintf(stderr, "[mali-repro] seq %zu tokens=%zu\n", i, inp.size());
        inputs.push_back(std::move(inp));
    }

    // Mirror the addon-side test driver: pack sequences into a batch, flushing
    // when batch.n_tokens + next-seq would exceed n_batch, identical to
    // examples/embedding/embedding.cpp:290. The Mali DeviceLost fires on the
    // very first llama_decode submit, so any chunking strategy reaches the bug.
    llama_batch batch = llama_batch_init(a.batch_size, /*embd*/ 0, /*n_seq_max*/ a.n_sequences);
    llama_memory_clear(llama_get_memory(ctx), true);

    const int n_embd = llama_model_n_embd(model);
    bool ok = true;
    int submit_idx = 0;
    int seq_in_batch = 0;
    int total_seqs_decoded = 0;
    auto flush = [&]() {
        if (batch.n_tokens == 0) return true;
        fprintf(stderr,
                "[mali-repro] submit #%d: llama_decode with %d tokens / %d sequences\n",
                submit_idx, batch.n_tokens, seq_in_batch);
        const int rc = llama_decode(ctx, batch);
        if (rc < 0) {
            fprintf(stderr,
                    "[mali-repro] FAIL: llama_decode returned %d on submit #%d "
                    "(this is the Mali Vulkan ErrorDeviceLost reproducer hit)\n",
                    rc, submit_idx);
            return false;
        }
        // Verify each sequence in this batch produced a finite embedding
        // before clearing memory for the next chunk.
        for (int s = 0; s < seq_in_batch; ++s) {
            const float * embd = llama_get_embeddings_seq(ctx, s);
            if (embd == nullptr) {
                fprintf(stderr, "[mali-repro] FAIL: no embedding for seq %d in submit #%d\n",
                        s, submit_idx);
                return false;
            }
            if (any_nan_or_inf(embd, n_embd)) {
                fprintf(stderr, "[mali-repro] FAIL: NaN/Inf in embedding for seq %d in submit #%d\n",
                        s, submit_idx);
                return false;
            }
        }
        total_seqs_decoded += seq_in_batch;
        ++submit_idx;
        llama_memory_clear(llama_get_memory(ctx), true);
        return true;
    };

    for (size_t s = 0; s < inputs.size(); ++s) {
        const auto & inp = inputs[s];
        if (batch.n_tokens + (int) inp.size() > a.batch_size || seq_in_batch >= a.n_sequences) {
            if (!flush()) { ok = false; break; }
            common_batch_clear(batch);
            seq_in_batch = 0;
        }
        for (size_t i = 0; i < inp.size(); ++i) {
            common_batch_add(batch, inp[i], (llama_pos) i, { (llama_seq_id) seq_in_batch }, /*logits*/ true);
        }
        ++seq_in_batch;
    }
    if (ok) ok = flush();

    if (ok) {
        fprintf(stderr,
                "[mali-repro] PASS: %d submits, %d sequences, all embeddings finite\n",
                submit_idx, total_seqs_decoded);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
