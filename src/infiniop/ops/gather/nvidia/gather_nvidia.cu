#include "gather_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include <cuda_runtime.h>

namespace op::gather::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    if (_opaque) delete _opaque;
}

__device__ __forceinline__ size_t prod_range_dev(const size_t *shape, int start, int end) {
    size_t p = 1;
    for (int i = start; i < end; ++i) p *= shape[i];
    return p;
}

template <typename TData, typename TIdx, int MAX_NDIM>
__global__ void gather_kernel(
    TData *output,
    const TData *input,
    const TIdx *index,
    size_t num_out,
    int ndim,
    int dim,
    const size_t *out_shape,
    const size_t *in_shape,
    const ptrdiff_t *in_strides) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t inner = prod_range_dev(out_shape, dim + 1, ndim);
    size_t dim_size_out = out_shape[dim];
    size_t dim_size_in = in_shape[dim];

    for (size_t t = tid; t < num_out; t += stride) {
        TIdx g = index[t];
        if (g < 0 || (size_t)g >= dim_size_in) {
            output[t] = (TData)0;
            continue;
        }

        size_t inner_idx = inner == 0 ? 0 : (t % inner);
        size_t tmp = inner == 0 ? t : (t / inner);

        size_t outer_flat = dim_size_out == 0 ? tmp : (tmp / dim_size_out);

        ptrdiff_t in_offset = 0;

        // prefix dims
        size_t rem = outer_flat;
        for (int k = dim - 1; k >= 0; --k) {
            size_t sz = out_shape[k];
            size_t coord = (sz == 0) ? 0 : (rem % sz);
            rem = (sz == 0) ? 0 : (rem / sz);
            in_offset += (ptrdiff_t)coord * in_strides[k];
            if (k == 0) break;
        }

        // dim
        in_offset += (ptrdiff_t)g * in_strides[dim];

        // suffix dims
        size_t rem2 = inner_idx;
        for (int k = ndim - 1; k >= dim + 1; --k) {
            size_t sz = out_shape[k];
            size_t coord = (sz == 0) ? 0 : (rem2 % sz);
            rem2 = (sz == 0) ? 0 : (rem2 / sz);
            in_offset += (ptrdiff_t)coord * in_strides[k];
        }

        output[t] = input[in_offset];
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t idx_desc,
    int dim) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info_result = GatherInfo::create(out_desc, in_desc, idx_desc, dim);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
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

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // copy small meta to device (simple & safe MVP)
    // NOTE: for performance you can cache these in Opaque later.
    int ndim = _info._ndim;
    int dim = _info._dim;
    size_t num_out = _info._num_out;

    // allocate on device via cudaMallocAsync? use plain cudaMalloc for MVP
    size_t *d_out_shape = nullptr;
    size_t *d_in_shape = nullptr;
    ptrdiff_t *d_in_strides = nullptr;

    cudaMalloc(&d_out_shape, sizeof(size_t) * ndim);
    cudaMalloc(&d_in_shape, sizeof(size_t) * ndim);
    cudaMalloc(&d_in_strides, sizeof(ptrdiff_t) * ndim);

    cudaMemcpyAsync(d_out_shape, _info._out_shape.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(d_in_shape, _info._in_shape.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(d_in_strides, _info._in_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, cuda_stream);

    int block = 256;
    int grid = (int)((num_out + block - 1) / block);
    grid = grid > 4096 ? 4096 : grid;

    auto dtype = _info.dtype();
    auto idx_dtype = _info.idx_dtype();

#define LAUNCH(TDATA)                                                                 \
    if (idx_dtype == INFINI_DTYPE_I32) {                                              \
        gather_kernel<TDATA, int32_t, 8><<<grid, block, 0, cuda_stream>>>(            \
            (TDATA *)output, (const TDATA *)input, (const int32_t *)index,            \
            num_out, ndim, dim, d_out_shape, d_in_shape, d_in_strides);               \
    } else if (idx_dtype == INFINI_DTYPE_I64) {                                       \
        gather_kernel<TDATA, int64_t, 8><<<grid, block, 0, cuda_stream>>>(            \
            (TDATA *)output, (const TDATA *)input, (const int64_t *)index,            \
            num_out, ndim, dim, d_out_shape, d_in_shape, d_in_strides);               \
    } else {                                                                         \
        cudaFree(d_out_shape); cudaFree(d_in_shape); cudaFree(d_in_strides);          \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                                       \
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        LAUNCH(half);
        break;
    case INFINI_DTYPE_BF16:
        LAUNCH(__nv_bfloat16);
        break;
    case INFINI_DTYPE_F32:
        LAUNCH(float);
        break;
    case INFINI_DTYPE_F64:
        LAUNCH(double);
        break;
    // integers: treat as storage types
    case INFINI_DTYPE_I8:  LAUNCH(int8_t); break;
    case INFINI_DTYPE_U8:  LAUNCH(uint8_t); break;
    case INFINI_DTYPE_I16: LAUNCH(int16_t); break;
    case INFINI_DTYPE_U16: LAUNCH(uint16_t); break;
    case INFINI_DTYPE_I32: LAUNCH(int32_t); break;
    case INFINI_DTYPE_U32: LAUNCH(uint32_t); break;
    case INFINI_DTYPE_I64: LAUNCH(int64_t); break;
    case INFINI_DTYPE_U64: LAUNCH(uint64_t); break;
    default:
        cudaFree(d_out_shape); cudaFree(d_in_shape); cudaFree(d_in_strides);
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH

    cudaFree(d_out_shape);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gather::nvidia