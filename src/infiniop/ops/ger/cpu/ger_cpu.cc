#include "ger_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::ger::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t A_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = GerInfo::createGerInfo(alpha_desc, x_desc, y_desc, A_desc);
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
infiniStatus_t calculateGer(
    const GerInfo &info,
    const Tdata *alpha,
    const Tdata *x,
    const Tdata *y,
    Tdata *A) {

    const size_t m = info.m;
    const size_t n = info.n;
    const ptrdiff_t incx = info.incx;
    const ptrdiff_t incy = info.incy;
    const ptrdiff_t A_row_stride = info.A_row_stride;
    const ptrdiff_t A_col_stride = info.A_col_stride;

    const Tdata alpha_v = alpha[0];
    for (size_t i = 0; i < m; ++i) {
        const Tdata temp = alpha_v * x[i * incx];
        for (size_t j = 0; j < n; ++j) {
            const ptrdiff_t A_idx = utils::cast<ptrdiff_t>(i) * A_row_stride + utils::cast<ptrdiff_t>(j) * A_col_stride;
            A[A_idx] += temp * y[j * incy];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_GER(TDATA)           \
    calculateGer(_info,                \
                 (const TDATA *)alpha, \
                 (const TDATA *)x,     \
                 (const TDATA *)y,     \
                 (TDATA *)A)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    const void *y,
    void *A,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        return CALCULATE_GER(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_GER(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_GER

} // namespace op::ger::cpu
