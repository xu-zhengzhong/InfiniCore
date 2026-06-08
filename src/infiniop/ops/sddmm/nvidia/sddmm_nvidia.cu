#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "sddmm_nvidia.cuh"

namespace op::sddmm::nvidia {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopSpMatDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->valuesDesc()->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32);

    auto result = SDDMMInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(
        dtype,
        c_desc->crowIndicesDesc()->dtype(),
        result.take(),
        c_desc,
        0,
        new Opaque(),
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tindex>
INFINIOP_CUDA_KERNEL sddmmKernel(
    size_t m,
    size_t k,
    ptrdiff_t a_row_stride,
    ptrdiff_t a_col_stride,
    ptrdiff_t b_row_stride,
    ptrdiff_t b_col_stride,
    const Tindex *crow,
    const Tindex *col,
    const float *a,
    const float *b,
    float *c,
    float alpha,
    float beta) {
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    for (Tindex ptr = crow[row]; ptr < crow[row + 1]; ++ptr) {
        auto c_col = static_cast<size_t>(col[ptr]);
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
            acc += a[row * a_row_stride + kk * a_col_stride] * b[kk * b_row_stride + c_col * b_col_stride];
        }
        c[ptr] = alpha * acc + beta * c[ptr];
    }
}

template <typename Tindex>
static infiniStatus_t launch(const SDDMMInfo &info, infiniopSpMatDescriptor_t c_desc, void *c_values, const void *a, const void *b, float alpha, float beta, void *stream) {
    constexpr size_t block = 128;
    auto grid = (info.m + block - 1) / block;
    sddmmKernel<Tindex><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        info.m,
        info.k,
        info.a_matrix.row_stride,
        info.a_matrix.col_stride,
        info.b_matrix.row_stride,
        info.b_matrix.col_stride,
        reinterpret_cast<const Tindex *>(c_desc->crowIndices()),
        reinterpret_cast<const Tindex *>(c_desc->colIndices()),
        reinterpret_cast<const float *>(a),
        reinterpret_cast<const float *>(b),
        reinterpret_cast<float *>(c_values),
        alpha,
        beta);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c_values,
    const void *a,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    switch (_index_dtype) {
    case INFINI_DTYPE_I32:
        return launch<int32_t>(_info, _c_desc, c_values, a, b, alpha, beta, stream);
    case INFINI_DTYPE_I64:
        return launch<int64_t>(_info, _c_desc, c_values, a, b, alpha, beta, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::sddmm::nvidia
