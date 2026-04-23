#include "../../../../utils.h"
#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"

#include "../cuda/kernel.cuh"
#include "asum_moore.h"

namespace op::asum::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t result_desc) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto info = AsumInfo::createAsumInfo(x_desc, result_desc);
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
infiniStatus_t calculateAsum(
    const AsumInfo &info,
    const Tdata *x,
    Tdata *result,
    musaStream_t stream) {

    const size_t size = info.getSize();
    const ptrdiff_t incx = info.getIncx();

    cuda::asum_kernel<256, Tdata, Tdata, Tcompute>
        <<<1, 256, 0, stream>>>(
            size,
            x,
            incx,
            result);

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_ASUM(TDATA, TCOMPUTE)                \
    calculateAsum<TDATA, TCOMPUTE>(_info,              \
                                   (const TDATA *)x,   \
                                   (TDATA *)result,    \
                                   (musaStream_t)stream)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *result,
    void *stream) const {

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return CALCULATE_ASUM(half, float);
    case INFINI_DTYPE_BF16:
        return CALCULATE_ASUM(cuda_bfloat16, float);
    case INFINI_DTYPE_F32:
        return CALCULATE_ASUM(float, float);
    case INFINI_DTYPE_F64:
        return CALCULATE_ASUM(double, double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

#undef CALCULATE_ASUM

} // namespace op::asum::moore