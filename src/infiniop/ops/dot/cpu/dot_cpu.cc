#include "dot_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::dot::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = DotInfo::createDotInfo(x_desc, y_desc);
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
infiniStatus_t calculateDot(
    const DotInfo &info,
    const void *x,
    const void *y,
    void *result) {

    const Tdata *x_ptr = reinterpret_cast<const Tdata *>(x);
    const Tdata *y_ptr = reinterpret_cast<const Tdata *>(y);
    Tdata *result_ptr = reinterpret_cast<Tdata *>(result);

    const ptrdiff_t n = static_cast<ptrdiff_t>(info.getSize());
    const ptrdiff_t incx = info.getIncx();
    const ptrdiff_t incy = info.getIncy();

    Tdata total = static_cast<Tdata>(0);
    ptrdiff_t ix = (incx < 0) ? (1 - n) * incx : 0;
    ptrdiff_t iy = (incy < 0) ? (1 - n) * incy : 0;

    for (ptrdiff_t i = 0; i < n; ++i) {
        total += x_ptr[ix] * y_ptr[iy];
        ix += incx;
        iy += incy;
    }

    result_ptr[0] = total;

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    const void *y,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F32:
        return calculateDot<float>(_info, x, y, result);
    case INFINI_DTYPE_F64:
        return calculateDot<double>(_info, x, y, result);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::dot::cpu