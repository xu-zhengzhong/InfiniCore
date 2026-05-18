#include "symv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::symv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SymvInfo::createSymvInfo(uplo, alpha_desc, A_desc, x_desc, beta_desc, y_desc);
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
infiniStatus_t calculateSymv(
    const SymvInfo &info,
    const Tdata *alpha,
    const Tdata *A,
    const Tdata *x,
    const Tdata *beta,
    Tdata *y) {

    const auto n = info.n;
    const auto row_stride = info.A_row_stride;
    const auto col_stride = info.A_col_stride;
    const auto incx = info.incx;
    const auto incy = info.incy;
    const auto alpha_v = alpha[0];
    const auto beta_v = beta[0];

    for (size_t i = 0; i < n; ++i) {
        const auto y_idx = utils::cast<ptrdiff_t>(i) * incy;
        if (beta_v == static_cast<Tdata>(0)) {
            y[y_idx] = static_cast<Tdata>(0);
        } else if (beta_v != static_cast<Tdata>(1)) {
            y[y_idx] = beta_v * y[y_idx];
        }
    }

    if (alpha_v == static_cast<Tdata>(0)) {
        return INFINI_STATUS_SUCCESS;
    }

    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto temp1 = alpha_v * xj;
            Tdata temp2 = static_cast<Tdata>(0);
            for (size_t i = 0; i < j; ++i) {
                const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                const auto y_idx = utils::cast<ptrdiff_t>(i) * incy;
                const auto A_v = A[A_idx];
                y[y_idx] += temp1 * A_v;
                temp2 += A_v * x[utils::cast<ptrdiff_t>(i) * incx];
            }
            const auto diag_idx = utils::cast<ptrdiff_t>(j) * (row_stride + col_stride);
            const auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += temp1 * A[diag_idx] + alpha_v * temp2;
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto temp1 = alpha_v * xj;
            Tdata temp2 = static_cast<Tdata>(0);
            const auto diag_idx = utils::cast<ptrdiff_t>(j) * (row_stride + col_stride);
            auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += temp1 * A[diag_idx];
            for (size_t i = j + 1; i < n; ++i) {
                const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                const auto A_v = A[A_idx];
                y_idx = utils::cast<ptrdiff_t>(i) * incy;
                y[y_idx] += temp1 * A_v;
                temp2 += A_v * x[utils::cast<ptrdiff_t>(i) * incx];
            }
            y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += alpha_v * temp2;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SYMV(TDATA)           \
    calculateSymv(_info,                \
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
        return CALCULATE_SYMV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SYMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SYMV

} // namespace op::symv::cpu
