#include "syr2_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::syr2::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t A_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = Syr2Info::createSyr2Info(uplo, alpha_desc, x_desc, y_desc, A_desc);
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
infiniStatus_t calculateSyr2(
    const Syr2Info &info,
    const Tdata *alpha,
    const Tdata *x,
    const Tdata *y,
    Tdata *A) {

    const auto n = info.n;
    const auto incx = info.incx;
    const auto incy = info.incy;
    const auto row_stride = info.A_row_stride;
    const auto col_stride = info.A_col_stride;
    const auto alpha_v = alpha[0];

    if (alpha_v == static_cast<Tdata>(0)) {
        return INFINI_STATUS_SUCCESS;
    }

    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto yj = y[utils::cast<ptrdiff_t>(j) * incy];
            if (xj != static_cast<Tdata>(0) || yj != static_cast<Tdata>(0)) {
                const auto temp1 = alpha_v * yj;
                const auto temp2 = alpha_v * xj;
                for (size_t i = 0; i <= j; ++i) {
                    const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                    A[A_idx] += x[utils::cast<ptrdiff_t>(i) * incx] * temp1 + y[utils::cast<ptrdiff_t>(i) * incy] * temp2;
                }
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto yj = y[utils::cast<ptrdiff_t>(j) * incy];
            if (xj != static_cast<Tdata>(0) || yj != static_cast<Tdata>(0)) {
                const auto temp1 = alpha_v * yj;
                const auto temp2 = alpha_v * xj;
                for (size_t i = j; i < n; ++i) {
                    const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                    A[A_idx] += x[utils::cast<ptrdiff_t>(i) * incx] * temp1 + y[utils::cast<ptrdiff_t>(i) * incy] * temp2;
                }
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SYR2(TDATA)           \
    calculateSyr2(_info,                \
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
        return CALCULATE_SYR2(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SYR2(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SYR2

} // namespace op::syr2::cpu
