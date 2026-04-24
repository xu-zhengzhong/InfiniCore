#include "../../../../utils.h"
#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"

#include "../cuda/kernel.cuh"
#include "scal_moore.h"

namespace op::scal::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
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

template <typename Tdata, typename Tcompute>
infiniStatus_t calculateScal(
    const ScalInfo &info,
    const Tdata *alpha,
    Tdata *x,
    musaStream_t stream) {

    const size_t size = info.getSize();
    const ptrdiff_t incx = info.getIncx();

    cuda::scal_kernel<256, Tdata, Tcompute>
        <<<1, 256, 0, stream>>>(
            size,
            alpha,
            x,
            incx);

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SCAL(TDATA, TCOMPUTE)                  \
    calculateScal<TDATA, TCOMPUTE>(_info,                \
                                   (const TDATA *)alpha, \
                                   (TDATA *)x,           \
                                   (musaStream_t)stream)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    void *x_desc,
    void *stream) const {

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return CALCULATE_SCAL(half, float);
    case INFINI_DTYPE_BF16:
        return CALCULATE_SCAL(cuda_bfloat16, float);
    case INFINI_DTYPE_F32:
        return CALCULATE_SCAL(float, float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SCAL(double, double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

#undef CALCULATE_SCAL

} // namespace op::scal::moore
