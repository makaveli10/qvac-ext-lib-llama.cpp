#include "ggml-impl.h"
#include "ggml-nnapi.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <cstring>
#include <map>
#include <tuple>

#include <android/NeuralNetworks.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <ggml-quants.h>

static OperandCode ggml_to_nnapi_type(ggml_type gt) {
    switch(gt) {
        case GGML_TYPE_F32:
            return ANEURALNETWORKS_TENSOR_FLOAT32;
        case GGML_TYPE_F16:
            return ANEURALNETWORKS_TENSOR_FLOAT16;
        case GGML_TYPE_Q8_0:
            return ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED;
        default:
            GGML_LOG_ERROR("Unsupported type %s", ggml_type_name(gt));
            assert(false);
    }
    return ANEURALNETWORKS_TENSOR_FLOAT32;
}

static std::pair<int, ANeuralNetworksMemory*> create_shared_memory(
        const char* name, uint32_t num_elements, size_t element_size, int prot) {
    int fd = ASharedMemory_create(name, num_elements * element_size);
    ANeuralNetworksMemory* memory = nullptr;
    int32_t status = ANeuralNetworksMemory_createFromFd(num_elements * element_size,
                                                        prot, fd, 0, &memory);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksMemory_createFromFd failed for %s", name);
        close(fd);
        return {-1, nullptr};
    }

    return {fd, memory};
}

static const char *operand_code_str (OperandCode code);

class nnapi_tensor {
public:
    std::vector<uint32_t> dimensions;
    ANeuralNetworksOperandType op_type = {};
    ANeuralNetworksMemory* memory = nullptr;
    int fd = -1;
    int64_t nels = 0;
    size_t size = 0;
    bool flipped_dimensions = false;
    std::string name;

    nnapi_tensor() = default;

    nnapi_tensor(const ggml_tensor * tensor, ggml_type pipeline_type,
                 bool flip_dimensions=false, bool is_output=false) {
        flipped_dimensions = flip_dimensions;
        name = std::string(tensor->name);
        if (flip_dimensions) {
            dimensions = {
                    1, // batch dimension
                    1, // batch dimension
                    static_cast<uint32_t>(tensor->ne[1]),
                    static_cast<uint32_t>(tensor->ne[0]),
            };
        } else {
            dimensions = {
                    1, // batch dimension
                    1, // batch dimension
                    static_cast<uint32_t>(tensor->ne[0]),
                    static_cast<uint32_t>(tensor->ne[1]),
            };
        }

        op_type = {
            .type = ggml_to_nnapi_type(pipeline_type),
            .dimensionCount = static_cast<uint32_t>(dimensions.size()),
            .dimensions = dimensions.data(),
            .scale = 0.0f,
            .zeroPoint = 0,
        };

        nels = ggml_nelements(tensor);

        size_t element_size = 0;
        switch (pipeline_type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
                element_size = ggml_type_size(pipeline_type);
                break;
            case GGML_TYPE_Q8_0:
                element_size = sizeof(int8_t);
                break;
            default:
                GGML_LOG_ERROR("Unsupported tensor size.");
        }
        size = nels * element_size;

        int prot = PROT_READ;
        if (is_output) {
            prot |= PROT_WRITE;
        }
        std::tie(fd, memory) = create_shared_memory(tensor->name,
                                                    static_cast<uint32_t>(nels),
                                                    element_size, prot);
    }

    void write(const ggml_tensor * tensor) const {
        switch(tensor->type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            {
                void* map = mmap(nullptr, size, PROT_WRITE, MAP_SHARED, fd, 0);
                memcpy(map, tensor->data, size);
                munmap(map, size);
                break;
            }
            case GGML_TYPE_Q8_0:
            {
                const uint8_t *ggml_raw = reinterpret_cast<uint8_t*>(tensor->data);
                void* nnapi_map = mmap(nullptr, size, PROT_WRITE, MAP_SHARED, fd, 0);
                auto *nnapi_target_int8 = reinterpret_cast<int8_t*>(nnapi_map);
                constexpr size_t type_size = sizeof(block_q8_0);
                constexpr size_t blck_size = QK8_0;
                const size_t nbytes = ggml_nbytes(tensor);
                const size_t nblocks = nbytes / type_size;

                size_t index_linear = 0;
                for (size_t current_block = 0; current_block < nblocks; current_block++) {
                    const size_t offset = current_block * type_size;
                    const uint8_t *block_start = ggml_raw + offset;
                    const auto *block = reinterpret_cast<const block_q8_0*>(block_start);

                    ggml_half block_delta = block->d;
                    const float ggml_block_scale = GGML_FP16_TO_FP32(block_delta);

                    for (int8_t q : block->qs) {
                        float dequantized = static_cast<float>(q) * ggml_block_scale;
                        auto requantized = static_cast<int8_t>(dequantized / op_type.scale);
                        nnapi_target_int8[index_linear] = requantized;
                        index_linear++;
                    }
                }
                munmap(nnapi_map, size);
                break;
            }
            default:
                GGML_LOG_ERROR("Unsupported type %s", ggml_type_name(tensor->type));
                assert(false);
        }
    }

    void write_transposed(const ggml_tensor * tensor) const {
        if (!(tensor->type == GGML_TYPE_F16 ||
              tensor->type == GGML_TYPE_F32 ||
              tensor->type == GGML_TYPE_Q8_0)) {
            GGML_LOG_ERROR("Unsupported type %s", ggml_type_name(tensor->type));
            assert(false);
        }

        uint32_t index_linear = 0;
        const uint8_t *data = reinterpret_cast<uint8_t*>(tensor->data);
        void* map = mmap(nullptr, size, PROT_WRITE, MAP_SHARED, fd, 0);
        auto* map_int8 = reinterpret_cast<int8_t*>(map);

        if (tensor->type == GGML_TYPE_Q8_0) {
            constexpr size_t type_size = sizeof(block_q8_0);
            constexpr size_t blck_size = QK8_0;
            const size_t nbytes = ggml_nbytes(tensor);
            const size_t nblocks = nbytes / type_size;
            const size_t blocks_per_width = tensor->ne[0] / QK8_0;

            for (size_t current_block = 0; current_block < nblocks; current_block++) {
                const size_t offset = current_block * type_size;
                const uint8_t *block_start = data + offset;
                const auto *block = reinterpret_cast<const block_q8_0*>(block_start);
                const float block_scale = GGML_FP16_TO_FP32(block->d);
                for (size_t j = 0; j < blck_size; ++j) {
                    size_t current_row = current_block / blocks_per_width;
                    size_t current_column = (current_block % blocks_per_width) * blck_size + j;
                    size_t index_transposed = current_column * tensor->ne[1] + current_row;
                    float dequantized = static_cast<float>(block->qs[j])*block_scale;
                    map_int8[index_transposed] = static_cast<int8_t>(dequantized / op_type.scale);
                }
            }
        } else if (flipped_dimensions && tensor->type == GGML_TYPE_F32) {
            const uint64_t n_elements = ggml_nelements(tensor);
            const float * src_data = reinterpret_cast<const float *>(tensor->data);

            if (op_type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
                auto * dst_data = reinterpret_cast<int8_t*>(map);
                const float inv_scale = 1.0f / op_type.scale;

                for (uint64_t i = 0; i < n_elements; ++i) {
                    float val = src_data[i] * inv_scale;
                    dst_data[i] = static_cast<int8_t>(std::clamp(std::round(val), -128.0f, 127.0f));
                }
            } else if (op_type.type == ANEURALNETWORKS_TENSOR_FLOAT32) {
                 memcpy(map, src_data, size);
            } else if (op_type.type == ANEURALNETWORKS_TENSOR_FLOAT16) {
                auto * dst_data = reinterpret_cast<_Float16*>(map);
                for (uint64_t i = 0; i < n_elements; ++i) {
                    dst_data[i] = static_cast<_Float16>(src_data[i]);
                }
            }
        }  else {
            for (int64_t i00 = 0; i00 < tensor->ne[0]; i00++) {
                for (int64_t i01 = 0; i01 < tensor->ne[1]; i01++) {
                    size_t index_transposed = i00 * tensor->nb[0] + i01 * tensor->nb[1];
                    if (tensor->type == GGML_TYPE_F32) {
                        if (op_type.type == ANEURALNETWORKS_TENSOR_FLOAT32) {
                            reinterpret_cast<float*>(map)[index_linear] = *reinterpret_cast<const float *>(&data[index_transposed]);
                        } else if (op_type.type == ANEURALNETWORKS_TENSOR_FLOAT16) {
                            reinterpret_cast<_Float16*>(map)[index_linear] = static_cast<_Float16>(*reinterpret_cast<const float *>(&data[index_transposed]));
                        } else if (op_type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
                            float unquantized = *reinterpret_cast<const float *>(&data[index_transposed]);
                            reinterpret_cast<int8_t*>(map)[index_linear] = static_cast<int8_t>(unquantized / op_type.scale);
                        } else {
                            // TODO: Not supported
                            assert(false);
                        }
                    } else if (tensor->type == GGML_TYPE_F16) {
                        reinterpret_cast<_Float16*>(map)[index_linear] = *reinterpret_cast<const _Float16 *>(&data[index_transposed]);
                    }
                    index_linear++;
                }
            }
        }
        munmap(map, size);
    }

    void read_transposed(ggml_tensor * tensor) const {
        // output is always f32 in the tests?
        if (tensor->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("Unsupported output tensor type %s", ggml_type_name(tensor->type));
            return;
        }

        if (op_type.type != ANEURALNETWORKS_TENSOR_FLOAT32 &&
            op_type.type != ANEURALNETWORKS_TENSOR_FLOAT16 &&
            op_type.type != ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
            GGML_LOG_ERROR("Unsupported type %d", op_type.type);
            return;
        }

        uint32_t index_linear = 0;
        void *map = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);

        if (op_type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
            auto *dst_float = reinterpret_cast<float*>(tensor->data);
            auto *nnapi_src = reinterpret_cast<int8_t*>(map);

            // TODO: Make this more similar to float
            for (int64_t i01 = 0; i01 < tensor->ne[1]; i01++) {
                for (int64_t i00 = 0; i00 < tensor->ne[0]; i00++) {
                    size_t index_transposed = i01 + i00 * tensor->ne[1];
                    dst_float[index_linear] = static_cast<float>(nnapi_src[index_transposed]) * op_type.scale;
                    index_linear++;
                }
            }
        } else {
            auto *dst_data = reinterpret_cast<uint8_t*>(tensor->data);
            for (int64_t i00 = 0; i00 < tensor->ne[0]; i00++) {
                for (int64_t i01 = 0; i01 < tensor->ne[1]; i01++) {
                    size_t index_transposed = i00 * tensor->nb[0] + i01 * tensor->nb[1];
                    if (op_type.type == ANEURALNETWORKS_TENSOR_FLOAT32) {
                        *reinterpret_cast<float *>(&dst_data[index_transposed]) = reinterpret_cast<float*>(map)[index_linear];
                    } else if (op_type.type == ANEURALNETWORKS_TENSOR_FLOAT16) {
                        *reinterpret_cast<float *>(&dst_data[index_transposed]) = reinterpret_cast<_Float16*>(map)[index_linear];
                    }
                    index_linear++;
                }
            }
        }

        munmap(map, size);
    }
};

static float tensor_get_max_scale(const ggml_tensor * tensor);
static float estimate_q80_output_scale(const ggml_tensor * src0, float scale0, float scale1);
static bool build_mat_mul_model(ANeuralNetworksModel** model,
                                ANeuralNetworksOperandType *in_tensor0_type,
                                ANeuralNetworksOperandType *in_tensor1_type,
                                ANeuralNetworksOperandType *out_tensor_type,
                                bool adj_y_value);
static bool compile_model(ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation);

struct nnapi_pipeline {
    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksCompilation* compilation = nullptr;

    nnapi_tensor src0;
    nnapi_tensor src1;
    nnapi_tensor dst;

    nnapi_pipeline(const struct ggml_tensor * a,
                   const struct ggml_tensor * b,
                   const struct ggml_tensor * c) {
        src0 = nnapi_tensor(a, a->type, true, false);
        if (src0.op_type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
            src0.op_type.scale = tensor_get_max_scale(a);
        }

        bool transpose_src1 = (b->type == GGML_TYPE_F32);
        src1 = nnapi_tensor(b, a->type, transpose_src1, false);
        if (src1.op_type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
            src1.op_type.scale = tensor_get_max_scale(b);
        }

        // Set NNAPI output dtype to match input, we will be converting to f32 later
        dst = nnapi_tensor(c, a->type, false, true);
        dst.op_type.type = ggml_to_nnapi_type(a->type);
        if (dst.op_type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
            dst.op_type.scale = estimate_q80_output_scale(a, src0.op_type.scale, src1.op_type.scale);
        }

//        GGML_LOG_ERROR("%s: %d x %d (%ld -> %ld). scale %f",
//                       operand_code_str((OperandCode)src0.op_type.type),
//                       src0.dimensions[2], src0.dimensions[3], src0.nels, src0.size,
//                       src0.op_type.scale);
//        GGML_LOG_ERROR("%s: %d x %d (%ld -> %ld). scale %f",
//                       operand_code_str((OperandCode)src1.op_type.type),
//                       src1.dimensions[2], src1.dimensions[3], src1.nels, src1.size,
//                       src1.op_type.scale);
//        GGML_LOG_ERROR("%s: %d x %d (%ld -> %ld). scale %f",
//                       operand_code_str((OperandCode)dst.op_type.type),
//                       dst.dimensions[2], dst.dimensions[3], dst.nels, dst.size,
//                       dst.op_type.scale);

        if (!build_mat_mul_model(&model,
                                 &src0.op_type,
                                 &src1.op_type,
                                 &dst.op_type,
                                 transpose_src1)) {
            GGML_LOG_ERROR("Failed to build the mat mul model");
            return;
        }

        if (!compile_model(model, &compilation)) {
            GGML_LOG_ERROR("Failed to compile model.");
            return;
        }
    }

    ~nnapi_pipeline() {
        ANeuralNetworksMemory_free(src0.memory);
        ANeuralNetworksMemory_free(src1.memory);
        ANeuralNetworksMemory_free(dst.memory);

        close(src0.fd);
        close(src1.fd);
        close(dst.fd);

        ANeuralNetworksCompilation_free(compilation);
        ANeuralNetworksModel_free(model);
    }
};

struct ggml_backend_nnapi_context {
    std::vector<ANeuralNetworksDevice*> devices;

    std::map<std::tuple<uint32_t, // N
                        uint32_t, // K
                        uint32_t, // M
                        ggml_type>,
             std::unique_ptr<nnapi_pipeline>> pipelines;
};

#define ENUM_TO_STR(r)                                                         \
  case r:                                                                      \
    return #r

static const char *
feature_leveL_code_str (FeatureLevelCode code)
{
    switch (code)
    {
        ENUM_TO_STR(ANEURALNETWORKS_FEATURE_LEVEL_1);
        ENUM_TO_STR(ANEURALNETWORKS_FEATURE_LEVEL_2);
        ENUM_TO_STR(ANEURALNETWORKS_FEATURE_LEVEL_3);
        ENUM_TO_STR(ANEURALNETWORKS_FEATURE_LEVEL_4);
        ENUM_TO_STR(ANEURALNETWORKS_FEATURE_LEVEL_5);
        ENUM_TO_STR(ANEURALNETWORKS_FEATURE_LEVEL_6);
        ENUM_TO_STR(ANEURALNETWORKS_FEATURE_LEVEL_7);
        ENUM_TO_STR(ANEURALNETWORKS_FEATURE_LEVEL_8);
        default:
            return "UNKNOWN FEATURE LEVEL";
    }
}

static const char *
device_type_code_str (DeviceTypeCode code)
{
    switch (code)
    {
        ENUM_TO_STR(ANEURALNETWORKS_DEVICE_UNKNOWN);
        ENUM_TO_STR(ANEURALNETWORKS_DEVICE_OTHER);
        ENUM_TO_STR(ANEURALNETWORKS_DEVICE_CPU);
        ENUM_TO_STR(ANEURALNETWORKS_DEVICE_GPU);
        ENUM_TO_STR(ANEURALNETWORKS_DEVICE_ACCELERATOR);
        default:
            return "UNKNOWN DEVICE TYPE";
    }
}

static const char *
operand_code_str (OperandCode code)
{
    switch (code)
    {
        ENUM_TO_STR(ANEURALNETWORKS_TENSOR_FLOAT16);
        ENUM_TO_STR(ANEURALNETWORKS_TENSOR_FLOAT32);
        ENUM_TO_STR(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED);
        ENUM_TO_STR(ANEURALNETWORKS_TENSOR_INT32);
        default:
            return "UNKNOWN OPERAND CODE";
    }
}

static void print_runtime_infos(ggml_backend_nnapi_context * ctx) {
    auto runtime_feature_level = static_cast<FeatureLevelCode>(ANeuralNetworks_getRuntimeFeatureLevel());
    GGML_LOG_INFO("Runtime feature level: %s", feature_leveL_code_str(runtime_feature_level));

    uint32_t num_devices = 0;
    int ret = ANeuralNetworks_getDeviceCount(&num_devices);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("Failed to get device count.");
        return;
    }

    GGML_LOG_INFO("Have %d devices:", num_devices);

    for (uint32_t i = 0; i < num_devices; i++) {
        ANeuralNetworksDevice* device;
        ret = ANeuralNetworks_getDevice(i, &device);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            GGML_LOG_ERROR("Failed to get device %d", i);
            return;
        }

        ctx->devices.push_back(device);

        int64_t device_feature_level_int;
        ret = ANeuralNetworksDevice_getFeatureLevel(device, &device_feature_level_int);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            GGML_LOG_ERROR("Failed to ANeuralNetworksDevice_getFeatureLevel for device %d", i);
            return;
        }
        auto device_feature_level = static_cast<FeatureLevelCode>(device_feature_level_int);

        const char* version;
        ret = ANeuralNetworksDevice_getVersion(device, &version);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            GGML_LOG_ERROR("Failed to ANeuralNetworksDevice_getVersion for device %d", i);
            return;
        }

        int32_t device_type_int;
        ret = ANeuralNetworksDevice_getType(device, &device_type_int);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            GGML_LOG_ERROR("Failed to ANeuralNetworksDevice_getType for device %d", i);
            return;
        }
        auto device_type = static_cast<DeviceTypeCode>(device_type_int);

        const char* name;
        ret = ANeuralNetworksDevice_getName(device, &name);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            GGML_LOG_ERROR("Failed to ANeuralNetworksDevice_getName for device %d", i);
            return;
        }
        GGML_LOG_INFO("Device %d: %s", i, name);
        GGML_LOG_INFO("       Type:    %s", device_type_code_str(device_type));
        GGML_LOG_INFO("       Version: %s", version);
        GGML_LOG_INFO("       Level:   %s", feature_leveL_code_str(device_feature_level));
    }
}

static void print_tensor_info(const char* name, const ggml_tensor * tensor) {

    int64_t nels = ggml_nelements(tensor);
    size_t row_size = ggml_row_size(tensor->type, nels);
    int64_t nrows = ggml_nrows(tensor);
    size_t nbytes = ggml_nbytes(tensor);

    GGML_LOG_ERROR("%s (%s): op %s (%s) %ldx%ldx%ldx%ld = %ld",
                   name, tensor->name, ggml_op_name(tensor->op), ggml_op_symbol(tensor->op),
                   tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], nels);
    GGML_LOG_ERROR("row_size %ld bytes nrows %ld nbytes %ld", row_size, nrows, nbytes);
    GGML_LOG_ERROR("+ type %s (%ld bytes) (%ld bytes per blck)",
                   ggml_type_name(tensor->type),
                   ggml_type_size(tensor->type),
                   ggml_blck_size(tensor->type));
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        GGML_LOG_ERROR("+ %d: elements %ld stride %ld bytes",
                       i, tensor->ne[i], tensor->nb[i]);
    }
}

static void print_ggml_f32_tensor(const ggml_tensor * tensor, bool shorten=true) {
    std::stringstream ss;

    constexpr uint32_t short_print_count = 4;

    auto *data = reinterpret_cast<float*>(tensor->data);

    GGML_LOG_ERROR("🍄 ggml f32 %ld x %ld %s:", tensor->ne[0], tensor->ne[1], tensor->name);

    if (tensor->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("Wrong type %s", ggml_type_name(tensor->type));
        return;
    }

    for (int64_t i01 = 0; i01 < tensor->ne[1]; i01++) {
        if (shorten && i01 >= short_print_count && i01 < tensor->ne[1] - short_print_count) {
            if (i01 == short_print_count) {
                GGML_LOG_INFO("[...],");
            }
            continue;
        }

        for (int64_t i00 = 0; i00 < tensor->ne[0]; i00++) {
            if (shorten && i00 >= short_print_count && i00 < tensor->ne[0] - short_print_count) {
                if (i00 == short_print_count) {
                    ss << "..., ";
                }
                continue;
            }
//            size_t index = i01 * tensor->ne[1] + i00;
//            ss << data[index];

            const void *x = (char *) tensor->data
                            + i00 * tensor->nb[0]
                            + i01 * tensor->nb[1];
            const auto *the_float = static_cast<const float*>(x);
            ss << *the_float;

            if (i00 < tensor->ne[0] - 1) {
                ss << ", ";
            }
        }
        GGML_LOG_INFO("[%s],", ss.str().c_str());
        ss.str("");
        ss.clear();
    }
}

static void print_ggml_q80_tensor(const ggml_tensor * tensor, bool print_quantized=false,
                                  bool shorten=true) {
    std::stringstream ss;

    constexpr uint32_t short_print_count = 4;

    size_t nbytes = ggml_nbytes(tensor);
    size_t type_size = ggml_type_size(tensor->type);
    size_t blck_size = ggml_blck_size(tensor->type); // == QK8_0
    size_t nblocks = nbytes / type_size;

    const size_t blocks_per_width = tensor->ne[0] / QK8_0;

    GGML_LOG_ERROR("🍄 ggml q8_0 %ld x %ld %s:", tensor->ne[0], tensor->ne[1], tensor->name);

    if (tensor->type != GGML_TYPE_Q8_0) {
        GGML_LOG_ERROR("Wrong type %s", ggml_type_name(tensor->type));
        return;
    }

    for (size_t current_block = 0; current_block < nblocks; current_block++) {
        size_t offset = current_block * type_size;
        const uint8_t *block_start = reinterpret_cast<uint8_t*>(tensor->data) + offset;
        const auto *block = reinterpret_cast<const block_q8_0*>(block_start);

        size_t current_row = current_block / blocks_per_width;

        if (shorten && current_row >= short_print_count && current_row < (size_t)tensor->ne[1] - short_print_count) {
            if (current_row == short_print_count && current_block == current_row * blocks_per_width) {
                GGML_LOG_INFO("[...],");
            }
            continue;
        }

        const float block_scale = GGML_FP16_TO_FP32(block->d);
        for (size_t j = 0; j < blck_size; ++j) {
            size_t current_column = (current_block % blocks_per_width) * blck_size + j;
            if (shorten && current_column >= short_print_count && current_column < (size_t)tensor->ne[0] - short_print_count) {
                if (current_column == short_print_count) {
                    ss << "..., ";
                }
                continue;
            }
            float dequantized = static_cast<float>(block->qs[j]) * block_scale;
            if (print_quantized) {
                ss << static_cast<int>(block->qs[j]);
            } else {
                ss << dequantized;
            }
            if (j < blck_size - 1) {
                ss << ", ";
            }
        }

        size_t next_row = (current_block + 1) / blocks_per_width;
        if (next_row > current_row) {
            GGML_LOG_INFO("[%s],", ss.str().c_str());
            ss.str("");
            ss.clear();
        }
    }
}

static void print_nnapi_q80_tensor(const nnapi_tensor * tensor, bool print_quantized=false,
                                   bool shorten=true) {
    std::stringstream ss;

    constexpr uint32_t short_print_count = 4;

    void* nnapi_map = mmap(nullptr, tensor->size, PROT_WRITE, MAP_SHARED, tensor->fd, 0);
    auto* nnapi_map_int8 = reinterpret_cast<int8_t*>(nnapi_map);

    uint32_t N = 0;
    uint32_t M = 0;
    if (tensor->flipped_dimensions) {
        N = tensor->dimensions[3];
        M = tensor->dimensions[2];
    } else {
        N = tensor->dimensions[2];
        M = tensor->dimensions[3];
    }

    GGML_LOG_ERROR("🍄 nnapi q8_0 %d x %d %s:", N, M, tensor->name.c_str());
    for (size_t x = 0; x < N; x++) {
        if (shorten && x >= short_print_count && x < N - short_print_count) {
            if (x == short_print_count) {
                GGML_LOG_INFO("[...],");
            }
            continue;
        }

        for (size_t y = 0; y < M; ++y) {
            if (shorten && y >= short_print_count && y < M - short_print_count) {
                if (y == short_print_count) {
                    ss << "..., ";
                }
                continue;
            }

            size_t index = x * M + y;
            if (print_quantized) {
                ss << static_cast<int>(nnapi_map_int8[index]);
            } else {
                float dequantized = static_cast<float>(nnapi_map_int8[index]) * tensor->op_type.scale;
                ss << dequantized;
            }
            if (y < M - 1) {
                ss << ", ";
            }
        }

        GGML_LOG_INFO("[%s],", ss.str().c_str());
        ss.str("");
        ss.clear();
    }

    munmap(nnapi_map, tensor->size);
}

static bool build_mat_mul_model(ANeuralNetworksModel** model,
                                ANeuralNetworksOperandType *in_tensor0_type,
                                ANeuralNetworksOperandType *in_tensor1_type,
                                ANeuralNetworksOperandType *out_tensor_type,
                                bool transpose_b) {

//    const uint32_t m = in_tensor0_type->dimensions[2];
//    const uint32_t n = in_tensor1_type->dimensions[3];
//    const uint32_t k = in_tensor1_type->dimensions[2];
//    GGML_LOG_ERROR("Building model: %d x %d @ %d x %d -> %d x %d (%d, %d, %d)",
//                   in_tensor0_type->dimensions[2], in_tensor0_type->dimensions[3],
//                   in_tensor1_type->dimensions[2], in_tensor1_type->dimensions[3],
//                   out_tensor_type->dimensions[2], out_tensor_type->dimensions[3],
//                   m, n, k);

    int ret = ANeuralNetworksModel_create(model);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksModel_create failed");
        return false;
    }

    uint32_t op_idx = 0;

    ANeuralNetworksOperandType scalarBoolType{
            .type = ANEURALNETWORKS_BOOL,
            .dimensionCount = 0,
            .dimensions = nullptr,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    ret = ANeuralNetworksModel_addOperand(*model, &scalarBoolType);
    uint32_t adj_x = op_idx++;
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksModel_addOperand failed for operand (%d)",
                       adj_x);
        return false;
    }
    bool adj_x_value = false;
    ret = ANeuralNetworksModel_setOperandValue(
            *model, (int32_t) adj_x, &adj_x_value,
            sizeof(adj_x_value));
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksModel_setOperandValue failed for operand (%d)",
                       adj_x);
        return false;
    }

    ret = ANeuralNetworksModel_addOperand(*model, &scalarBoolType);
    uint32_t adj_y = op_idx++;
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksModel_addOperand failed for operand (%d)",
                       adj_y);
        return false;
    }
    bool adj_y_value = transpose_b;
    ret = ANeuralNetworksModel_setOperandValue(
        *model, (int32_t) adj_y, &adj_y_value,
        sizeof(adj_y_value));
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksModel_setOperandValue failed for operand (%d)",
                       adj_y);
        return false;
    }

    uint32_t tensor0_in = op_idx++;

    ret = ANeuralNetworksModel_addOperand(*model, in_tensor0_type);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("addOperand failed for tensor0_in operand of type %s",
                       operand_code_str((OperandCode)in_tensor0_type->type));
        return false;
    }

    uint32_t tensor1_in = op_idx++;

    ret = ANeuralNetworksModel_addOperand(*model, in_tensor1_type);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("addOperand failed for tensor1_in operand of type %s",
                       operand_code_str((OperandCode)in_tensor1_type->type));
        return false;
    }

    uint32_t tensor_out = op_idx++;

    ret = ANeuralNetworksModel_addOperand(*model, out_tensor_type);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("addOperand failed for tensor_out operand of type %s",
                       operand_code_str((OperandCode)out_tensor_type->type));
        return false;
    }

    // Add the BATCH_MATMUL operation.
    std::vector<uint32_t> mulInputOperands = {
            tensor0_in,
            tensor1_in,
            adj_x,
            adj_y
    };
    ret = ANeuralNetworksModel_addOperation(
            *model, ANEURALNETWORKS_BATCH_MATMUL, mulInputOperands.size(),
            mulInputOperands.data(), 1, &tensor_out);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksModel_addOperation failed for BATCH_MATMUL");
        return false;
    }

    std::vector<uint32_t> modelInputs = {
            tensor0_in,
            tensor1_in,
    };
    std::vector<uint32_t> modelOutputs = {
            tensor_out,
    };
    ret = ANeuralNetworksModel_identifyInputsAndOutputs(
            *model, modelInputs.size(), modelInputs.data(), modelOutputs.size(),
            modelOutputs.data());
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksModel_identifyInputsAndOutputs failed");
        return false;
    }

    ret = ANeuralNetworksModel_finish(*model);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksModel_finish failed");
        return false;
    }

    return true;
}

static void check_device_support_for_model(ggml_backend_nnapi_context * ctx,
                                           ANeuralNetworksModel* model) {
    for (ANeuralNetworksDevice* device : ctx->devices) {

        const char* name;
        int ret = ANeuralNetworksDevice_getName(device, &name);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            GGML_LOG_ERROR("Failed to ANeuralNetworksDevice_getName for device %p", (void*) device);
            continue;
        }

        bool is_first_op_supported = false;

        ret = ANeuralNetworksModel_getSupportedOperationsForDevices(model, &device,
                                                                    1, &is_first_op_supported);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            GGML_LOG_ERROR("ANeuralNetworksModel_getSupportedOperationsForDevices failed");
            continue;
        }

        ANeuralNetworksCompilation* compilation = nullptr;
        const ANeuralNetworksDevice* devices[] = { device };
        ret = ANeuralNetworksCompilation_createForDevices(model, devices, 1, &compilation);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            GGML_LOG_ERROR("ANeuralNetworksCompilation_createForDevices failed");
            GGML_LOG_INFO("%s: supported %d", name, false);
            continue;
        }
        ANeuralNetworksCompilation_free(compilation);

        GGML_LOG_INFO("%s: supported %d", name, is_first_op_supported);
    }
}

static bool compile_model(ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation) {
    // ANeuralNetworksCompilation_createForDevices
    int ret = ANeuralNetworksCompilation_create(model, compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksCompilation_create failed");
        return false;
    }

    ret = ANeuralNetworksCompilation_setPreference(
            *compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksCompilation_setPreference failed");
        return false;
    }

    ret = ANeuralNetworksCompilation_finish(*compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksCompilation_finish failed");
        return false;
    }

    return true;
}

static bool dispatch_model(ANeuralNetworksCompilation* compilation,
                           nnapi_tensor *in_tensor0,
                           nnapi_tensor *in_tensor1,
                           nnapi_tensor *out_tensor) {
    ANeuralNetworksEvent* event = nullptr;
    ANeuralNetworksExecution* execution;
    int ret = ANeuralNetworksExecution_create(compilation, &execution);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksExecution_create failed");
        return false;
    }

    ret = ANeuralNetworksExecution_setInputFromMemory(
            execution, 0, &in_tensor0->op_type, in_tensor0->memory, 0, in_tensor0->size);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksExecution_setInputFromMemory failed for in_tensor0");
        return false;
    }

    ret = ANeuralNetworksExecution_setInputFromMemory(
            execution, 1, &in_tensor1->op_type, in_tensor1->memory, 0, in_tensor1->size);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksExecution_setInputFromMemory failed for in_tensor1");
        return false;
    }

    ret = ANeuralNetworksExecution_setOutputFromMemory(
            execution, 0, &out_tensor->op_type, out_tensor->memory, 0, out_tensor->size);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksExecution_setOutputFromMemory failed for out_tensor0");
        return false;
    }

    const ANeuralNetworksEvent* const* dependencies = nullptr;
    uint32_t numDependencies = 0;

    ret = ANeuralNetworksExecution_startComputeWithDependencies(
            execution, dependencies, numDependencies, 0, &event);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        GGML_LOG_ERROR("ANeuralNetworksExecution_compute failed");
        return false;
    }

    ANeuralNetworksEvent_wait(event);
    ANeuralNetworksExecution_free(execution);
    ANeuralNetworksEvent_free(event);

    return true;
}

static float tensor_get_max_scale(const ggml_tensor * tensor) {
    const uint8_t *ggml_raw = reinterpret_cast<uint8_t*>(tensor->data);

    switch (tensor->type) {
        case GGML_TYPE_Q8_0: {
            constexpr size_t type_size = sizeof(block_q8_0);
            const size_t nblocks = ggml_nbytes(tensor) / type_size;
            float max_block_scale = std::numeric_limits<float>::min();

            for (size_t current_block = 0; current_block < nblocks; current_block++) {
                const auto *block = reinterpret_cast<const block_q8_0*>(ggml_raw + current_block * type_size);
                max_block_scale = std::max(max_block_scale, GGML_FP16_TO_FP32(block->d));
            }
            return max_block_scale;
        }
        case GGML_TYPE_F32: {
            const float *data_f32 = reinterpret_cast<float*>(tensor->data);
            float max_value = *std::max_element(data_f32, data_f32 + ggml_nelements(tensor));
            return max_value / 127.0f;
        }
        case GGML_TYPE_F16: {
            const _Float16 *data_f16 = reinterpret_cast<_Float16*>(tensor->data);
            float max_value = *std::max_element(data_f16, data_f16 + ggml_nelements(tensor));
            return max_value / 127.0f;
        }
        default:
            GGML_LOG_ERROR("Unsupported type %s", ggml_type_name(tensor->type));
            assert(false);
    }
}

// TODO: This was determined by running NxM tests
static float estimate_q80_output_scale(const ggml_tensor * src0, float scale0, float scale1) {
    const auto K = static_cast<float>(src0->ne[0]);
    const float max_int8_abs = 127.0f;
    float max_abs_output = K * max_int8_abs * std::max(scale0, scale1);

    constexpr float clearance = 1.7f; // TODO: this currently needs to be larger for larger Ks
    float estimated_abs_max = clearance * std::sqrt(max_abs_output);
//    GGML_LOG_ERROR("estimated abs max output %f", estimated_abs_max);

    return estimated_abs_max / max_int8_abs;
}

static void ggml_backend_nnapi_mul_mat(ggml_backend_nnapi_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const auto op_tuple = std::tuple<uint32_t, uint32_t, uint32_t, ggml_type>(src0->ne[1],
                                                                              src0->ne[0],
                                                                              src1->ne[1],
                                                                              src0->type);
    // TODO: do this even earlier
    if (!ctx->pipelines.count(op_tuple)) {
//        const int64_t m = src0->ne[1];
//        const int64_t n = src1->ne[1];
//        const int64_t k = src1->ne[0];
//        GGML_LOG_ERROR("Building pipeline for: %ld x %ld @ %ld x %ld -> %ld x %ld (%ld, %ld, %ld)",
//                       m, k, k, n, m, n, m, n, k);

        auto p = std::make_unique<nnapi_pipeline>(src0, src1, dst);
        ctx->pipelines.emplace(op_tuple, std::move(p));
    }
    nnapi_pipeline *pipeline = ctx->pipelines.at(op_tuple).get();

//    if (src0->type == GGML_TYPE_F32) {
//        print_ggml_f32_tensor(src0);
//    } else if (src0->type == GGML_TYPE_Q8_0) {
//        print_ggml_q80_tensor(src0, false, false);
//    }

//    if (src0->type == GGML_TYPE_F32) {
//        print_ggml_f32_tensor(src1);
//    } else if (src0->type == GGML_TYPE_Q8_0) {
//        print_ggml_q80_tensor(src1, false, false);
//    }

    pipeline->src0.write(src0);
    pipeline->src1.write_transposed(src1);

//    print_nnapi_q80_tensor(&pipeline->src0, false, false);
//    print_nnapi_q80_tensor(&pipeline->src1, false, false);

    if (!dispatch_model(pipeline->compilation,
                        &pipeline->src0,
                        &pipeline->src1,
                        &pipeline->dst)) {
        GGML_LOG_ERROR("Failed to dispatch model.");
        return;
    }

//    print_nnapi_q80_tensor(&pipeline->dst, false, false);

    pipeline->dst.read_transposed(dst);

//    print_ggml_f32_tensor(dst);
}

static void check_device_op_support(ggml_backend_nnapi_context * ctx, OperandCode type,
                                    uint32_t m, uint32_t n, uint32_t k) {
    uint32_t src0_d[] = {1, 1, m, k};
    ANeuralNetworksOperandType src0 = {
            .type = type,
            .dimensionCount = 4,
            .dimensions = src0_d,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    if (type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
        src0.scale = 1.0f;
    }

    uint32_t src1_d[] = {1, 1, k, n};
    ANeuralNetworksOperandType src1 = {
            .type = type,
            .dimensionCount = 4,
            .dimensions = src1_d,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    if (type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
        src1.scale = 1.0f;
    }

    uint32_t dst_d[] = {1, 1, m, n};
    ANeuralNetworksOperandType dst = {
            .type = type,
            .dimensionCount = 4,
            .dimensions = dst_d,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    if (type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED) {
        dst.scale = 1.0f;
    }

    ANeuralNetworksModel* model = nullptr;
    if (!build_mat_mul_model(&model, &src0, &src1, &dst, false)) {
        GGML_LOG_ERROR("Failed to build the mat mul model for type %s",
                       operand_code_str(type));
    }

    GGML_LOG_INFO("%d x %d @ %d x %d -> %d x %d (%d, %d, %d) (%s)",
                  m, k, k, n, m, n, m, n, k, operand_code_str(type));
    check_device_support_for_model(ctx, model);
    ANeuralNetworksModel_free(model);
}

static void print_device_model_support(ggml_backend_nnapi_context * ctx) {
    // supported types for matmul
    std::vector<OperandCode> types_to_test = {
            ANEURALNETWORKS_TENSOR_FLOAT16,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED,
            ANEURALNETWORKS_TENSOR_INT32
    };

    for (OperandCode tensor_type_code : types_to_test) {
        check_device_op_support(ctx, tensor_type_code, 4, 4, 4);
    }

    check_device_op_support(ctx, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2048, 17, 1024);
}

// backend interface
static const char * ggml_backend_nnapi_get_name(ggml_backend_t backend) {
    return "NNAPI";

    GGML_UNUSED(backend);
}

static void ggml_backend_nnapi_free(ggml_backend_t backend) {
    auto * ctx = reinterpret_cast<ggml_backend_nnapi_context *>(backend->context);
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_nnapi_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = reinterpret_cast<ggml_backend_nnapi_context *>(backend->context);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_nnapi_mul_mat(ctx, node);
                break;
            case GGML_OP_OUT_PROD:
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i nnapi_backend_iface = {
        .get_name           = ggml_backend_nnapi_get_name,
        .free               = ggml_backend_nnapi_free,
        .set_tensor_async   = nullptr,
        .get_tensor_async   = nullptr,
        .cpy_tensor_async   = nullptr,
        .synchronize        = nullptr,
        .graph_plan_create  = nullptr,
        .graph_plan_free    = nullptr,
        .graph_plan_update  = nullptr,
        .graph_plan_compute = nullptr,
        .graph_compute      = ggml_backend_nnapi_graph_compute,
        .event_record       = nullptr,
        .event_wait         = nullptr,
        .optimize_graph     = nullptr,
};

static ggml_guid_t ggml_backend_nnapi_guid() {
    static ggml_guid guid = { 0xde, 0xad, 0xbe, 0xef,
                              0xde, 0xad, 0xbe, 0xef,
                              0xde, 0xad, 0xbe, 0xef,
                              0xde, 0xad, 0xbe, 0xef };
    return &guid;
}

ggml_backend_t ggml_backend_nnapi_init(void) {
    auto * ctx = new ggml_backend_nnapi_context;

    auto * backend = new ggml_backend {
        .guid    = ggml_backend_nnapi_guid(),
        .iface   = nnapi_backend_iface,
        .device  = ggml_backend_reg_dev_get(ggml_backend_nnapi_reg(), 0),
        .context = ctx,
    };

//    print_runtime_infos(ctx);
//    print_device_model_support(ctx);

    return backend;
}

bool ggml_backend_is_nnapi(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_nnapi_guid());
}

void ggml_backend_nnapi_set_n_threads(ggml_backend_t backend_nnapi, int n_threads) {
    GGML_ASSERT(ggml_backend_is_nnapi(backend_nnapi));
    GGML_UNUSED(n_threads);
}

// device interface
static const char * ggml_backend_nnapi_device_get_name(ggml_backend_dev_t dev) {
    return "NNAPI Device";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_nnapi_device_get_description(ggml_backend_dev_t dev) {
    return "NNAPI Device Description";

    GGML_UNUSED(dev);
}

static void ggml_backend_nnapi_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_nnapi_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_nnapi_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_nnapi_device_get_name(dev);
    props->description = ggml_backend_nnapi_device_get_description(dev);
    props->type        = ggml_backend_nnapi_device_get_type(dev);
    ggml_backend_nnapi_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        .async                = false,
        .host_buffer          = false,
        .buffer_from_host_ptr = true,
        .events               = false,
    };
}

static ggml_backend_t ggml_backend_nnapi_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_nnapi_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_nnapi_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_nnapi_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_nnapi_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {

//    GGML_LOG_WARN("Testing NNAPI device for tensor %s (op %s) on ctx %p",
//                  op->name, ggml_op_name(op->op), dev->context);

//    return false;

    switch (op->op) {
        case GGML_OP_MUL_MAT:
        {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

//            {
//                const int64_t m = src0->ne[1];
//                const int64_t n = src1->ne[1];
//                const int64_t k = src1->ne[0];
//                GGML_LOG_WARN("Want to run MUL_MAT: %ld x %ld @ %ld x %ld -> %ld x %ld (%ld, %ld, %ld)",
//                              m, k, k, n, m, n, m, n, k);
//                return false;
//            }


            // TODO: Implement heterogeneous input types
            if (src0->type != src1->type && src1->type != GGML_TYPE_F32) {
                return false;
            }

            // TODO: Implement more types
            switch(src0->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                    break;
                default:
                    return false;
            }

            // TODO: Implement batching
            if (src0->ne[2] != 1 ||
                src0->ne[3] != 1 ||
                src1->ne[2] != 1 ||
                src1->ne[3] != 1) {
                return false;
            }

            // TODO: Implement block partitioning
            if (src0->ne[0] > 4736 ||
                src0->ne[1] > 4736 ||
                src1->ne[0] > 4736 ||
                src1->ne[1] > 4736) {
                return false;
            }

            // TODO: Implement block partitioning
            constexpr uint32_t max_nels_in = 3200 * 3200;
            if (src0->ne[0] * src0->ne[0] > max_nels_in ||
                src1->ne[0] * src1->ne[1] > max_nels_in) {
                return false;
            }

            // TODO: Figure out max output
            constexpr uint32_t max_nels_out = 4096 * 4096;
            if (op->ne[0] * op->ne[1] > max_nels_out) {
                return false;
            }

//            print_tensor_info("src0", src0);
//            print_tensor_info("src1", src1);
//            print_tensor_info("dst", op);

//            GGML_LOG_WARN("Tensor 0 type: %s op: %s (%s) ne0 %ld cont %d",
//                          ggml_type_name(src0->type),
//                          ggml_op_name(src0->op),
//                          ggml_op_symbol(src0->op),
//                          src0->ne[0],
//                          ggml_is_contiguous(src0));
//            GGML_LOG_WARN("Tensor 1 type: %s op: %s (%s) ne0 %ld cont %d",
//                          ggml_type_name(src1->type),
//                          ggml_op_name(src1->op),
//                          ggml_op_symbol(src1->op),
//                          src0->ne[1],
//                          ggml_is_contiguous(src1));

//            const int64_t m = src0->ne[1];
//            const int64_t n = src1->ne[1];
//            const int64_t k = src1->ne[0];
//            GGML_LOG_WARN("Will run on NNAPI: %ld x %ld @ %ld x %ld -> %ld x %ld (%ld, %ld, %ld)",
//                          m, k, k, n, m, n, m, n, k);

            // TODO: Figure out if we can support non contiguous inputs
            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1);
        }
        case GGML_OP_NONE:
            return true;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_OUT_PROD:
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_nnapi_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_nnapi_device_i = {
    .get_name             = ggml_backend_nnapi_device_get_name,
    .get_description      = ggml_backend_nnapi_device_get_description,
    .get_memory           = ggml_backend_nnapi_device_get_memory,
    .get_type             = ggml_backend_nnapi_device_get_type,
    .get_props            = ggml_backend_nnapi_device_get_props,
    .init_backend         = ggml_backend_nnapi_device_init_backend,
    .get_buffer_type      = ggml_backend_nnapi_device_get_buffer_type,
    .get_host_buffer_type = nullptr,
    .buffer_from_host_ptr = ggml_backend_nnapi_device_buffer_from_host_ptr,
    .supports_op          = ggml_backend_nnapi_device_supports_op,
    .supports_buft        = ggml_backend_nnapi_device_supports_buft,
    .offload_op           = nullptr,
    .event_new            = nullptr,
    .event_free           = nullptr,
    .event_synchronize    = nullptr,
};

// backend reg interface
static const char * ggml_backend_nnapi_reg_get_name(ggml_backend_reg_t reg) {
    return "NNAPI";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_nnapi_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_nnapi_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_nnapi_device = {
        .iface   = ggml_backend_nnapi_device_i,
        .reg     = reg,
        .context = nullptr,
    };

    return &ggml_backend_nnapi_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_nnapi_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_nnapi_set_n_threads;
    }
    return nullptr;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_nnapi_reg_i = {
    .get_name         = ggml_backend_nnapi_reg_get_name,
    .get_device_count = ggml_backend_nnapi_reg_get_device_count,
    .get_device       = ggml_backend_nnapi_reg_get_device,
    .get_proc_address = ggml_backend_nnapi_get_proc_address,
};

ggml_backend_reg_t ggml_backend_nnapi_reg(void) {
    static struct ggml_backend_reg ggml_backend_nnapi_reg = {
        .api_version = GGML_BACKEND_API_VERSION,
        .iface       = ggml_backend_nnapi_reg_i,
        .context     = nullptr,
    };

    return &ggml_backend_nnapi_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_nnapi_reg)
