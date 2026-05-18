#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "spmm_nvidia.cuh"

namespace op::spmm::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopSpMatDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = SpMMInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        a_desc->crowIndicesDesc()->dtype(),
        result.take(),
        a_desc,
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
__device__ float loadAsFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ float loadAsFloat<half>(half value) {
    return __half2float(value);
}

template <>
__device__ float loadAsFloat<cuda_bfloat16>(cuda_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T storeFromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ half storeFromFloat<half>(float value) {
    return __float2half(value);
}

template <>
__device__ cuda_bfloat16 storeFromFloat<cuda_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename Tdata, typename Tindex>
__global__ void spmmKernel(
    SpMMInfo info,
    infiniopSpMatDescriptor_t a_desc,
    Tdata *c,
    const Tdata *b,
    float alpha,
    float beta) {
    auto linear = blockIdx.x * blockDim.x + threadIdx.x;
    auto total = info.m * info.n;
    if (linear >= total) {
        return;
    }

    auto row = linear / info.n;
    auto col = linear % info.n;
    auto values = reinterpret_cast<const Tdata *>(a_desc->values());
    auto crow_indices = reinterpret_cast<const Tindex *>(a_desc->crowIndices());
    auto col_indices = reinterpret_cast<const Tindex *>(a_desc->colIndices());

    float acc = 0;
    for (Tindex ptr = crow_indices[row]; ptr < crow_indices[row + 1]; ++ptr) {
        auto k = static_cast<size_t>(col_indices[ptr]);
        auto b_offset = k * info.b_matrix.row_stride + col * info.b_matrix.col_stride;
        acc += loadAsFloat(values[ptr]) * loadAsFloat(b[b_offset]);
    }

    auto c_offset = row * info.c_matrix.row_stride + col * info.c_matrix.col_stride;
    if (beta == 0) {
        c[c_offset] = storeFromFloat<Tdata>(alpha * acc);
    } else {
        c[c_offset] = storeFromFloat<Tdata>(alpha * acc + beta * loadAsFloat(c[c_offset]));
    }
}

template <typename Tdata, typename Tindex>
infiniStatus_t calculate(
    const SpMMInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *c,
    const void *b,
    float alpha,
    float beta,
    void *stream) {
    constexpr int block_size = 256;
    auto total = info.m * info.n;
    auto grid_size = std::min<size_t>((total + block_size - 1) / block_size, 65535);
    if (grid_size == 0) {
        grid_size = 1;
    }

    spmmKernel<Tdata, Tindex><<<grid_size, block_size, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        info,
        a_desc,
        reinterpret_cast<Tdata *>(c),
        reinterpret_cast<const Tdata *>(b),
        alpha,
        beta);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateByIndex(
    infiniDtype_t index_dtype,
    const SpMMInfo &info,
    infiniopSpMatDescriptor_t a_desc,
    void *c,
    const void *b,
    float alpha,
    float beta,
    void *stream) {
    switch (index_dtype) {
    case INFINI_DTYPE_I32:
        return calculate<Tdata, int32_t>(info, a_desc, c, b, alpha, beta, stream);
    case INFINI_DTYPE_I64:
        return calculate<Tdata, int64_t>(info, a_desc, c, b, alpha, beta, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return calculateByIndex<half>(_index_dtype, _info, _a_desc, c, b, alpha, beta, stream);
    case INFINI_DTYPE_BF16:
        return calculateByIndex<cuda_bfloat16>(_index_dtype, _info, _a_desc, c, b, alpha, beta, stream);
    case INFINI_DTYPE_F32:
        return calculateByIndex<float>(_index_dtype, _info, _a_desc, c, b, alpha, beta, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::spmm::nvidia
