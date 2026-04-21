#include "rot_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::rot::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = RotInfo::createRotInfo(x_desc, y_desc);
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
infiniStatus_t calculateRot(const RotInfo &info, void *x, void *y, const void *c, const void *s) {
    using Tcompute = std::conditional_t<std::is_same_v<Tdata, double>, double, float>;

    Tdata *x_ptr = reinterpret_cast<Tdata *>(x);
    Tdata *y_ptr = reinterpret_cast<Tdata *>(y);
    const Tcompute c_val = utils::cast<Tcompute>(reinterpret_cast<const Tdata *>(c)[0]);
    const Tcompute s_val = utils::cast<Tcompute>(reinterpret_cast<const Tdata *>(s)[0]);

    const ptrdiff_t size = static_cast<ptrdiff_t>(info.getSize());
    const ptrdiff_t incx = info.getIncx();
    const ptrdiff_t incy = info.getIncy();

    if (size <= 0) {
        return INFINI_STATUS_SUCCESS;
    }

    const ptrdiff_t ix = incx >= 0 ? 0 : (size - 1) * (-incx);
    const ptrdiff_t iy = incy >= 0 ? 0 : (size - 1) * (-incy);

#pragma omp parallel for if (size > 1024)
    for (ptrdiff_t i = 0; i < size; ++i) {
        const ptrdiff_t x_idx = ix + i * incx;
        const ptrdiff_t y_idx = iy + i * incy;

        const Tcompute x_val = utils::cast<Tcompute>(x_ptr[x_idx]);
        const Tcompute y_val = utils::cast<Tcompute>(y_ptr[y_idx]);
        const Tcompute temp = c_val * x_val + s_val * y_val;
        y_ptr[y_idx] = utils::cast<Tdata>(c_val * y_val - s_val * x_val);
        x_ptr[x_idx] = utils::cast<Tdata>(temp);
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *x,
    void *y,
    const void *c,
    const void *s,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return calculateRot<fp16_t>(_info, x, y, c, s);
    case INFINI_DTYPE_BF16:
        return calculateRot<bf16_t>(_info, x, y, c, s);
    case INFINI_DTYPE_F32:
        return calculateRot<float>(_info, x, y, c, s);
    case INFINI_DTYPE_F64:
        return calculateRot<double>(_info, x, y, c, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::rot::cpu