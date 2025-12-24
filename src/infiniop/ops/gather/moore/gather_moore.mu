#include "gather_moore.h"
#include "gather_moore_kernel.h"

#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_handle.h"

#include <musa_runtime.h>
#include <vector>

namespace op::gather::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;

    // device meta
    size_t *d_out_shape = nullptr;
    size_t *d_in_shape = nullptr;
    ptrdiff_t *d_in_strides = nullptr;
    int ndim = 0;
};

Descriptor::~Descriptor() {
    if (_opaque) {
        if (_opaque->d_out_shape) musaFree(_opaque->d_out_shape);
        if (_opaque->d_in_shape) musaFree(_opaque->d_in_shape);
        if (_opaque->d_in_strides) musaFree(_opaque->d_in_strides);
        delete _opaque;
    }
}

template <typename T, typename TIdx>
__global__ void gather_kernel(
    T *output,
    const T *input,
    const TIdx *index,
    size_t num_out,
    int ndim,
    int dim,
    const size_t *out_shape,
    const size_t *in_shape,
    const ptrdiff_t *in_strides) {

    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    GatherOp<T, TIdx> op;

    for (; t < num_out; t += stride) {
        op(t, output, input, index, num_out, ndim, dim, out_shape, in_shape, in_strides);
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t idx_desc,
    int dim) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto info_result = GatherInfo::create(out_desc, in_desc, idx_desc, dim);
    CHECK_RESULT(info_result);
    auto info = info_result.take();

    int ndim = info._ndim;

    auto opaque = new Opaque();
    opaque->internal = handle->internal();
    opaque->ndim = ndim;

    musaMalloc((void **)&opaque->d_out_shape, sizeof(size_t) * ndim);
    musaMalloc((void **)&opaque->d_in_shape, sizeof(size_t) * ndim);
    musaMalloc((void **)&opaque->d_in_strides, sizeof(ptrdiff_t) * ndim);

    musaMemcpy(opaque->d_out_shape, info._out_shape.data(), sizeof(size_t) * ndim, musaMemcpyHostToDevice);
    musaMemcpy(opaque->d_in_shape, info._in_shape.data(), sizeof(size_t) * ndim, musaMemcpyHostToDevice);
    musaMemcpy(opaque->d_in_strides, info._in_strides.data(), sizeof(ptrdiff_t) * ndim, musaMemcpyHostToDevice);

    *desc_ptr = new Descriptor(
        opaque,
        std::move(info),
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

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    size_t num_out = _info._num_out;
    int ndim = _opaque->ndim;
    int dim = _info._dim;

    int block = 256;
    int grid = (int)((num_out + block - 1) / block);
    if (grid > 4096) grid = 4096;

    auto dtype = _info.dtype();
    auto idx_dtype = _info.idx_dtype();

#define LAUNCH(T)                                                                    \
    do {                                                                             \
        if (idx_dtype == INFINI_DTYPE_I32) {                                         \
            gather_kernel<T, int32_t><<<grid, block, 0, musa_stream>>>(              \
                (T *)output, (const T *)input, (const int32_t *)index,               \
                num_out, ndim, dim,                                                  \
                _opaque->d_out_shape, _opaque->d_in_shape, _opaque->d_in_strides);   \
        } else if (idx_dtype == INFINI_DTYPE_I64) {                                  \
            gather_kernel<T, int64_t><<<grid, block, 0, musa_stream>>>(              \
                (T *)output, (const T *)input, (const int64_t *)index,               \
                num_out, ndim, dim,                                                  \
                _opaque->d_out_shape, _opaque->d_in_shape, _opaque->d_in_strides);   \
        } else {                                                                     \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                                   \
        }                                                                            \
    } while (0)

    switch (dtype) {
    case INFINI_DTYPE_F16:
        LAUNCH(half);
        break;
    case INFINI_DTYPE_BF16:
        LAUNCH(__mt_bfloat16);
        break;
    case INFINI_DTYPE_F32:
        LAUNCH(float);
        break;
    case INFINI_DTYPE_F64:
        LAUNCH(double);
        break;
    case INFINI_DTYPE_I8:
        LAUNCH(int8_t);
        break;
    case INFINI_DTYPE_U8:
        LAUNCH(uint8_t);
        break;
    case INFINI_DTYPE_I16:
        LAUNCH(int16_t);
        break;
    case INFINI_DTYPE_U16:
        LAUNCH(uint16_t);
        break;
    case INFINI_DTYPE_I32:
        LAUNCH(int32_t);
        break;
    case INFINI_DTYPE_U32:
        LAUNCH(uint32_t);
        break;
    case INFINI_DTYPE_I64:
        LAUNCH(int64_t);
        break;
    case INFINI_DTYPE_U64:
        LAUNCH(uint64_t);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gather::moore