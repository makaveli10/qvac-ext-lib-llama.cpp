// Reproducer for the Mali-G715 (Pixel 9 Pro) grammar regression observed by
// the qvac-lib-infer-llamacpp-llm addon when running Qwen3-0.6B-Q8_0 with a
// simple GBNF (`root ::= ("yes" | "no")`).
//
// On the affected Mali Vulkan build, `common_sampler_sample` selects a token
// whose decoded piece doesn't extend any active grammar stack (e.g. "@",
// token id 31). `llama_grammar_accept_token` then sees all stacks become
// empty and throws:
//
//   Unexpected empty grammar stack after accepting piece: @ (31)
//
// Same code path runs cleanly on:
//   - CPU (any device)
//   - Adreno 750 (Samsung Galaxy S25 Ultra) via OpenCL
//   - Desktop Vulkan / NVIDIA / Apple Metal
//
// Pass: every sampled token's decoded piece is a prefix of "yes" or "no"
//       (or EOG), and the loop exits cleanly with `out` ∈ {"yes", "no"}.
// Fail: any sampled token's piece contains a character not in "yesno", or
//       common_sampler_sample propagates the empty-stack runtime_error.
//
// Usage: pass the gguf path either as argv[1] or via $LLAMACPP_TEST_MODELFILE.
//        Optional --max-tokens <N> bounds how many tokens we try to sample
//        before failing with a "no terminal" diagnostic.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "get-model.h"
#include "llama.h"
#include "sampling.h"

namespace {

struct args {
    const char * model_path = nullptr;
    int max_tokens = 8;
};

args parse_args(int argc, char ** argv) {
    args a;
    a.model_path = get_model_or_exit(argc, argv);
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            a.max_tokens = std::atoi(argv[++i]);
        }
    }
    return a;
}

// Build the same prompt the addon-side test uses (Qwen3 chat template,
// system + user with /no_think to suppress reasoning, asking a yes/no
// question). We construct it manually so this test does not need to load
// the jinja templates from the GGUF.
std::string build_prompt() {
    return std::string(
        "<|im_start|>system\n"
        "You are a binary classifier. Reply only \"yes\" or \"no\". /no_think\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Is the sky blue on a clear day?\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n");
}

bool piece_is_prefix_of_yesno(const std::string & piece) {
    // Allow only chars that can legitimately appear in a token whose
    // decoded text is a prefix of "yes" or "no". Whitespace counts because
    // some BPE tokens carry a leading space marker.
    for (unsigned char c : piece) {
        switch (c) {
            case 'y': case 'e': case 's':
            case 'n': case 'o':
            case ' ': case '\t':
                continue;
            default:
                return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const args a = parse_args(argc, argv);

    fprintf(stderr, "[mali-grammar-repro] model=%s max_tokens=%d\n",
            a.model_path, a.max_tokens);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999; // matches the failing addon config
    llama_model * model = llama_model_load_from_file(a.model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "[mali-grammar-repro] failed to load model\n");
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 1024;
    cparams.n_batch   = 1024;
    cparams.n_ubatch  = 512;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "[mali-grammar-repro] failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    // Tokenize the prompt.
    const std::string prompt = build_prompt();
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, prompt, true, true);
    fprintf(stderr, "[mali-grammar-repro] prompt tokens=%zu\n", prompt_tokens.size());

    // Configure the sampler with the same GBNF the addon test uses.
    common_params_sampling sparams;
    sparams.seed    = 42;
    sparams.temp    = 1.0f;
    sparams.grammar = "root ::= (\"yes\" | \"no\")";

    common_sampler * smpl = common_sampler_init(model, sparams);
    if (smpl == nullptr) {
        fprintf(stderr, "[mali-grammar-repro] failed to init sampler with grammar\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    // Prefill the prompt in a single batch.
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "[mali-grammar-repro] prompt decode failed\n");
        common_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    bool ok = true;
    std::string out;

    for (int i = 0; i < a.max_tokens; ++i) {
        llama_token id = LLAMA_TOKEN_NULL;

        try {
            id = common_sampler_sample(smpl, ctx, -1, /*grammar_first*/ false);
            common_sampler_accept(smpl, id, /*accept_grammar*/ true);
        } catch (const std::exception & e) {
            fprintf(stderr,
                    "[mali-grammar-repro] FAIL: sampler/grammar threw at step %d: %s\n",
                    i, e.what());
            ok = false;
            break;
        }

        if (id == LLAMA_TOKEN_NULL) {
            fprintf(stderr, "[mali-grammar-repro] FAIL: sampler returned LLAMA_TOKEN_NULL at step %d\n", i);
            ok = false;
            break;
        }

        if (llama_vocab_is_eog(vocab, id)) {
            fprintf(stderr, "[mali-grammar-repro] EOG at step %d\n", i);
            break;
        }

        const std::string piece = common_token_to_piece(ctx, id, /*special*/ false);
        fprintf(stderr, "[mali-grammar-repro] step %d token=%d piece=\"%s\"\n",
                i, id, piece.c_str());

        if (!piece_is_prefix_of_yesno(piece)) {
            fprintf(stderr,
                    "[mali-grammar-repro] FAIL: grammar mask leaked — sampled piece \"%s\" (token %d) "
                    "is not a prefix of \"yes\" or \"no\"\n",
                    piece.c_str(), id);
            ok = false;
            break;
        }

        out += piece;

        // Feed the sampled token back so the next step has up-to-date logits.
        llama_batch step = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, step) < 0) {
            fprintf(stderr, "[mali-grammar-repro] FAIL: llama_decode failed for sampled token %d at step %d\n",
                    id, i);
            ok = false;
            break;
        }

        // Stop once we have a complete answer.
        std::string trimmed = out;
        while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\t')) {
            trimmed.erase(trimmed.begin());
        }
        while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\t' ||
                                    trimmed.back() == '\n' || trimmed.back() == '\r')) {
            trimmed.pop_back();
        }
        if (trimmed == "yes" || trimmed == "no") {
            fprintf(stderr, "[mali-grammar-repro] grammar terminated cleanly with \"%s\"\n",
                    trimmed.c_str());
            break;
        }
    }

    if (ok) {
        fprintf(stderr, "[mali-grammar-repro] PASS: produced \"%s\" (no grammar leak, no empty-stack throw)\n",
                out.c_str());
    }

    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
