#include "gemv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::gemv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasOperation_t trans,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = GemvInfo::createGemvInfo(trans, alpha_desc, A_desc, x_desc, beta_desc, y_desc);
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
infiniStatus_t calculateGemv(
    const GemvInfo &info,
    const Tdata *alpha,
    const Tdata *A,
    const Tdata *x,
    const Tdata *beta,
    Tdata *y) {

    const size_t m = info.trans == INFINIOP_BLAS_OP_N ? info.m : info.n;
    const size_t n = info.trans == INFINIOP_BLAS_OP_N ? info.n : info.m;
    const ptrdiff_t A_row_stride = info.A_row_stride;
    const ptrdiff_t A_col_stride = info.A_col_stride;
    const ptrdiff_t incx = info.incx;
    const ptrdiff_t incy = info.incy;

    const Tdata alpha_v = alpha[0];
    const Tdata beta_v = beta[0];
    for (size_t i = 0; i < m; ++i) {
        Tdata sum = static_cast<Tdata>(0);
        for (size_t j = 0; j < n; ++j) {
            const ptrdiff_t A_idx = info.trans == INFINIOP_BLAS_OP_N
                                      ? utils::cast<ptrdiff_t>(i) * A_row_stride + utils::cast<ptrdiff_t>(j) * A_col_stride
                                      : utils::cast<ptrdiff_t>(j) * A_row_stride + utils::cast<ptrdiff_t>(i) * A_col_stride;
            sum += A[A_idx] * x[utils::cast<ptrdiff_t>(j) * incx];
        }

        const ptrdiff_t y_idx = utils::cast<ptrdiff_t>(i) * incy;
        if (beta_v == static_cast<Tdata>(0)) {
            y[y_idx] = alpha_v * sum;
        } else {
            y[y_idx] = alpha_v * sum + beta_v * y[y_idx];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_GEMV(TDATA)           \
    calculateGemv(_info,                \
                  (const TDATA *)alpha, \
                  (const TDATA *)A,     \
                  (const TDATA *)x,     \
                  (const TDATA *)beta,  \
                  (TDATA *)y)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *A,
    const void *x,
    const void *beta,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        return CALCULATE_GEMV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_GEMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_GEMV

} // namespace op::gemv::cpu
