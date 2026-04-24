#include "axpy_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <type_traits>

namespace op::axpy::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = AxpyInfo::createAxpyInfo(alpha_desc, x_desc, y_desc);
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
infiniStatus_t calculateAxpy(
    const AxpyInfo &info,
    const void *alpha,
    const void *x,
    void *y) {

    const Tdata *alpha_ptr = reinterpret_cast<const Tdata *>(alpha);
    const Tdata *x_ptr = reinterpret_cast<const Tdata *>(x);
    Tdata *y_ptr = reinterpret_cast<Tdata *>(y);

    const ptrdiff_t size = info.getSize();
    const ptrdiff_t incx = info.getIncx();
    const ptrdiff_t incy = info.getIncy();

    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
        const float alpha_f = utils::cast<float>(alpha_ptr[0]);
        for (ptrdiff_t i = 0; i < size; ++i) {
            const float x_f = utils::cast<float>(x_ptr[i * incx]);
            const float y_f = utils::cast<float>(y_ptr[i * incy]);
            y_ptr[i * incy] = utils::cast<Tdata>(alpha_f * x_f + y_f);
        }
    } else {
        const Tdata alpha_v = alpha_ptr[0];
        for (ptrdiff_t i = 0; i < size; ++i) {
            y_ptr[i * incy] = alpha_v * x_ptr[i * incx] + y_ptr[i * incy];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return calculateAxpy<fp16_t>(_info, alpha, x, y);
    case INFINI_DTYPE_BF16:
        return calculateAxpy<bf16_t>(_info, alpha, x, y);
    case INFINI_DTYPE_F32:
        return calculateAxpy<float>(_info, alpha, x, y);
    case INFINI_DTYPE_F64:
        return calculateAxpy<double>(_info, alpha, x, y);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::axpy::cpu