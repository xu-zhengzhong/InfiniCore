#include "swap_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::swap::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = SwapInfo::createSwapInfo(x_desc, y_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateSwap(const SwapInfo &info, void *x, void *y) {
    Tdata *x_ptr = reinterpret_cast<Tdata *>(x);
    Tdata *y_ptr = reinterpret_cast<Tdata *>(y);

    const ptrdiff_t size = info.getSize();

#pragma omp parallel for if (size > 1024)
    for (ptrdiff_t i = 0; i < size; ++i) {
        size_t x_idx = i * info.getIncx();
        size_t y_idx = i * info.getIncy();
        Tdata temp = x_ptr[x_idx];
        x_ptr[x_idx] = y_ptr[y_idx];
        y_ptr[y_idx] = temp;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *x,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return calculateSwap<fp16_t>(_info, x, y);
    case INFINI_DTYPE_BF16:
        return calculateSwap<bf16_t>(_info, x, y);
    case INFINI_DTYPE_F32:
        return calculateSwap<float>(_info, x, y);
    case INFINI_DTYPE_F64:
        return calculateSwap<double>(_info, x, y);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::swap::cpu
