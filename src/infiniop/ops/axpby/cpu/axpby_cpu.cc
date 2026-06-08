#include "axpby_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::axpby::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = AxpbyInfo::create(x_desc, y_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
static void calculateAxpby(
    const AxpbyInfo &info,
    const void *x,
    void *y,
    float alpha,
    float beta) {
    auto x_data = reinterpret_cast<const Tdata *>(x);
    auto y_data = reinterpret_cast<Tdata *>(y);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(info.n); ++i) {
        auto x_offset = i * info.incx;
        auto y_offset = i * info.incy;
        auto result = alpha * utils::cast<float>(x_data[x_offset]) + beta * utils::cast<float>(y_data[y_offset]);
        y_data[y_offset] = utils::cast<Tdata>(result);
    }
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
    (void)stream;

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        calculateAxpby<fp16_t>(_info, x, y, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        calculateAxpby<bf16_t>(_info, x, y, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        calculateAxpby<float>(_info, x, y, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F64:
        calculateAxpby<double>(_info, x, y, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::axpby::cpu
