#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "axpby_nvidia.cuh"

#include <cuda_fp16.h>

namespace op::axpby::nvidia {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    CHECK_DTYPE(x_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    auto result = AxpbyInfo::create(x_desc, y_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        new Opaque(),
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
INFINIOP_CUDA_KERNEL axpbyKernel(
    size_t n,
    ptrdiff_t incx,
    ptrdiff_t incy,
    const Tdata *x,
    Tdata *y,
    float alpha,
    float beta) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    auto result = alpha * static_cast<float>(x[i * incx]) + beta * static_cast<float>(y[i * incy]);
    y[i * incy] = static_cast<Tdata>(result);
}

template <>
INFINIOP_CUDA_KERNEL axpbyKernel<__half>(
    size_t n,
    ptrdiff_t incx,
    ptrdiff_t incy,
    const __half *x,
    __half *y,
    float alpha,
    float beta) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    auto result = alpha * __half2float(x[i * incx]) + beta * __half2float(y[i * incy]);
    y[i * incy] = __float2half(result);
}

template <typename Tdata>
static infiniStatus_t launch(const AxpbyInfo &info, const void *x, void *y, float alpha, float beta, void *stream) {
    constexpr size_t block = 256;
    auto grid = (info.n + block - 1) / block;
    axpbyKernel<Tdata><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        info.n,
        info.incx,
        info.incy,
        reinterpret_cast<const Tdata *>(x),
        reinterpret_cast<Tdata *>(y),
        alpha,
        beta);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *y,
    float alpha,
    float beta,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<__half>(_info, x, y, alpha, beta, stream);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, x, y, alpha, beta, stream);
    case INFINI_DTYPE_F64:
        return launch<double>(_info, x, y, alpha, beta, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::axpby::nvidia
