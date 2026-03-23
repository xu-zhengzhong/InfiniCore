#include "scal_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::scal::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = ScalInfo::createScalInfo(y_desc, x_desc);
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
infiniStatus_t calculateScal(const ScalInfo &info, void *y, const void *x, float alpha) {
    Tdata *y_ptr = reinterpret_cast<Tdata *>(y);
    const Tdata *x_ptr = reinterpret_cast<const Tdata *>(x);

    const ptrdiff_t size = info.getSize();

#pragma omp parallel for if (size > 1024)
    for (ptrdiff_t i = 0; i < size; ++i) {
        size_t y_idx = info.isYContiguous()
                         ? i
                         : op::common_cpu::indexToOffset(i, info.getNdim(), info.getShape(), info.getYStrides());
        size_t x_idx = info.isXContiguous()
                         ? i
                         : op::common_cpu::indexToOffset(i, info.getNdim(), info.getShape(), info.getXStrides());

        if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
            y_ptr[y_idx] = utils::cast<Tdata>(utils::cast<float>(x_ptr[x_idx]) * utils::cast<float>(alpha));
        } else {
            y_ptr[y_idx] = x_ptr[x_idx] * alpha;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    float alpha,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F32:
        return calculateScal<float>(_info, y, x, alpha);
    case INFINI_DTYPE_F16:
        return calculateScal<fp16_t>(_info, y, x, alpha);
    case INFINI_DTYPE_BF16:
        return calculateScal<bf16_t>(_info, y, x, alpha);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::scal::cpu