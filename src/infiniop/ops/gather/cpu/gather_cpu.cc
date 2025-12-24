#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <omp.h>
#include <type_traits>

namespace op::gather::cpu {

Descriptor::~Descriptor() = default;

static inline size_t prod_range(const std::vector<size_t> &shape, int start, int end) {
    // [start, end)
    size_t p = 1;
    for (int i = start; i < end; ++i) p *= shape[i];
    return p;
}

template <typename TData, typename TIdx>
static void gather_cpu_impl(const GatherInfo &info, void *output, const void *input, const void *index) {
    const auto &out_shape = info._out_shape;
    const auto &in_shape = info._in_shape;
    const auto &in_strides = info._in_strides;
    int ndim = info._ndim;
    int dim = info._dim;

    // out/index are assumed contiguous in tests (out always contiguous).
    // We'll read index as contiguous by linear t. If index can be strided in future,
    // you can extend with idx_strides address computation.
    size_t num_out = info._num_out;
    size_t inner = prod_range(out_shape, dim + 1, ndim);
    size_t dim_size_out = out_shape[dim];
    size_t dim_size_in = in_shape[dim];

    auto out_ptr = reinterpret_cast<TData *>(output);
    auto in_ptr = reinterpret_cast<const TData *>(input);
    auto idx_ptr = reinterpret_cast<const TIdx *>(index);

#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_out; ++t) {
        TIdx g = idx_ptr[t];

        if (g < 0 || static_cast<size_t>(g) >= dim_size_in) {
            if constexpr (std::is_arithmetic_v<TData>) {
                out_ptr[t] = static_cast<TData>(0);
            } else {
                out_ptr[t] = utils::cast<TData>(0.0f);
            }
            continue;
        }

        // Decompose t into coordinates based on out_shape (contiguous order).
        // coords are used to compute input offset with in_strides, but at dim we use g.
        size_t inner_idx = inner == 0 ? 0 : (t % inner);
        size_t tmp = inner == 0 ? t : (t / inner);

        // tmp indexes the prefix [0..dim] part in a flattened manner where dim axis length = dim_size_out
        size_t dim_i = dim_size_out == 0 ? 0 : (tmp % dim_size_out);
        size_t outer_flat = dim_size_out == 0 ? tmp : (tmp / dim_size_out);

        // compute input offset = sum_k coord_k * in_stride[k], with coord_dim = g
        ptrdiff_t in_offset = 0;

        // prefix dims [0, dim)
        size_t rem = outer_flat;
        for (int k = dim - 1; k >= 0; --k) {
            size_t sz = out_shape[k];
            size_t coord = (sz == 0) ? 0 : (rem % sz);
            rem = (sz == 0) ? 0 : (rem / sz);
            in_offset += static_cast<ptrdiff_t>(coord) * in_strides[k];
            if (k == 0) break;
        }

        // dim
        (void)dim_i; // dim_i is output coordinate, but input coordinate is g
        in_offset += static_cast<ptrdiff_t>(g) * in_strides[dim];

        // suffix dims (dim+1..ndim)
        size_t rem2 = inner_idx;
        for (int k = ndim - 1; k >= dim + 1; --k) {
            size_t sz = out_shape[k];
            size_t coord = (sz == 0) ? 0 : (rem2 % sz);
            rem2 = (sz == 0) ? 0 : (rem2 / sz);
            in_offset += static_cast<ptrdiff_t>(coord) * in_strides[k];
        }

        out_ptr[t] = in_ptr[in_offset];
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t idx_desc,
    int dim) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info_result = GatherInfo::create(out_desc, in_desc, idx_desc, dim);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(
        nullptr,
        info_result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *index,
    void *stream) const {

    auto dtype = _info.dtype();
    auto idx_dtype = _info.idx_dtype();

#define DISPATCH_IDX(TDATA)                                                       \
    switch (idx_dtype) {                                                          \
    case INFINI_DTYPE_I32:                                                        \
        gather_cpu_impl<TDATA, int32_t>(_info, output, input, index);             \
        return INFINI_STATUS_SUCCESS;                                             \
    case INFINI_DTYPE_I64:                                                        \
        gather_cpu_impl<TDATA, int64_t>(_info, output, input, index);             \
        return INFINI_STATUS_SUCCESS;                                             \
    default:                                                                      \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                                    \
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        DISPATCH_IDX(fp16_t);
    case INFINI_DTYPE_BF16:
        DISPATCH_IDX(bf16_t);
    case INFINI_DTYPE_F32:
        DISPATCH_IDX(float);
    case INFINI_DTYPE_F64:
        DISPATCH_IDX(double);

    case INFINI_DTYPE_I8:
        DISPATCH_IDX(int8_t);
    case INFINI_DTYPE_U8:
        DISPATCH_IDX(uint8_t);
    case INFINI_DTYPE_I16:
        DISPATCH_IDX(int16_t);
    case INFINI_DTYPE_U16:
        DISPATCH_IDX(uint16_t);
    case INFINI_DTYPE_I32:
        DISPATCH_IDX(int32_t);
    case INFINI_DTYPE_U32:
        DISPATCH_IDX(uint32_t);
    case INFINI_DTYPE_I64:
        DISPATCH_IDX(int64_t);
    case INFINI_DTYPE_U64:
        DISPATCH_IDX(uint64_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef DISPATCH_IDX
}

} // namespace op::gather::cpu