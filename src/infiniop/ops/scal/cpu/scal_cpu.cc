#include "scal_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <type_traits>

namespace op::scal::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = ScalInfo::createScalInfo(alpha_desc, x_desc);
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
infiniStatus_t calculateScal(
    const ScalInfo &info,
    const void *alpha,
    void *x) {

    const Tdata *alpha_ptr = reinterpret_cast<const Tdata *>(alpha);
    Tdata *x_ptr = reinterpret_cast<Tdata *>(x);

    const ptrdiff_t size = info.getSize();
    const ptrdiff_t incx = info.getIncx();

    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
        const float alpha_f = utils::cast<float>(alpha_ptr[0]);
        for (ptrdiff_t i = 0; i < size; ++i) {
            const float x_f = utils::cast<float>(x_ptr[i * incx]);
            x_ptr[i * incx] = utils::cast<Tdata>(alpha_f * x_f);
        }
    } else {
        const Tdata alpha_v = alpha_ptr[0];
        for (ptrdiff_t i = 0; i < size; ++i) {
            x_ptr[i * incx] = alpha_v * x_ptr[i * incx];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    void *x,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return calculateScal<fp16_t>(_info, alpha, x);
    case INFINI_DTYPE_BF16:
        return calculateScal<bf16_t>(_info, alpha, x);
    case INFINI_DTYPE_F32:
        return calculateScal<float>(_info, alpha, x);
    case INFINI_DTYPE_F64:
        return calculateScal<double>(_info, alpha, x);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::scal::cpu