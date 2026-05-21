#include "sbmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::sbmv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    size_t k,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SbmvInfo::createSbmvInfo(uplo, k, alpha_desc, A_desc, x_desc, beta_desc, y_desc);
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
infiniStatus_t calculateSbmv(
    const SbmvInfo &info,
    const Tdata *alpha,
    const Tdata *A,
    const Tdata *x,
    const Tdata *beta,
    Tdata *y) {

    const auto n = info.n;
    const auto k = info.k;
    const auto lda = info.A_col_stride;
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
        const auto kplus1 = k + 1;
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto temp1 = alpha_v * xj;
            Tdata temp2 = static_cast<Tdata>(0);
            const auto i_begin = j > k ? j - k : 0;
            const auto l = utils::cast<ptrdiff_t>(kplus1) - utils::cast<ptrdiff_t>(j) - 1;
            for (size_t i = i_begin; i < j; ++i) {
                const auto A_idx = l + utils::cast<ptrdiff_t>(i) + utils::cast<ptrdiff_t>(j) * lda;
                const auto y_idx = utils::cast<ptrdiff_t>(i) * incy;
                const auto A_v = A[A_idx];
                y[y_idx] += temp1 * A_v;
                temp2 += A_v * x[utils::cast<ptrdiff_t>(i) * incx];
            }
            const auto diag_idx = utils::cast<ptrdiff_t>(k) + utils::cast<ptrdiff_t>(j) * lda;
            const auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += temp1 * A[diag_idx] + alpha_v * temp2;
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto temp1 = alpha_v * xj;
            Tdata temp2 = static_cast<Tdata>(0);
            auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += temp1 * A[utils::cast<ptrdiff_t>(j) * lda];

            const auto i_end = std::min(n, j + k + 1);
            const auto l = -utils::cast<ptrdiff_t>(j);
            for (size_t i = j + 1; i < i_end; ++i) {
                const auto A_idx = l + utils::cast<ptrdiff_t>(i) + utils::cast<ptrdiff_t>(j) * lda;
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

#define CALCULATE_SBMV(TDATA)           \
    calculateSbmv(_info,                \
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
        return CALCULATE_SBMV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SBMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SBMV

} // namespace op::sbmv::cpu
