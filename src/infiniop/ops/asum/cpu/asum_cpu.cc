#include "asum_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::asum::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = AsumInfo::createAsumInfo(x_desc);
    CHECK_RESULT(info);

    // Create descriptor
    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateAsum(
    const AsumInfo &info,
    const void *x,
    void *result) {

    const Tdata *x_ptr = reinterpret_cast<const Tdata *>(x);
    Tdata *result_ptr = reinterpret_cast<Tdata *>(result);

    const ptrdiff_t size = info.getSize();

    Tdata total_sum = 0.0;

    for (ptrdiff_t i = 0; i < size; ++i) {
        size_t idx = i * info.getIncx();
        total_sum += std::abs(x_ptr[idx]);
    }

    result_ptr[0] = total_sum;

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F32:
        return calculateAsum<float>(_info, x, result);
    case INFINI_DTYPE_F64:
        return calculateAsum<double>(_info, x, result);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::asum::cpu