#include "gbmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::gbmv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasOperation_t trans,
    size_t kl,
    size_t ku,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = GbmvInfo::createGbmvInfo(trans, kl, ku, alpha_desc, A_desc, x_desc, beta_desc, y_desc);
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
infiniStatus_t calculateGbmv(
    const GbmvInfo &info,
    const Tdata *alpha,
    const Tdata *A,
    const Tdata *x,
    const Tdata *beta,
    Tdata *y) {

    const auto m = info.m;
    const auto n = info.n;
    const auto kl = info.kl;
    const auto ku = info.ku;
    const auto lda = info.A_col_stride;
    const auto incx = info.incx;
    const auto incy = info.incy;
    const auto alpha_v = alpha[0];
    const auto beta_v = beta[0];
    const auto y_len = info.trans == INFINIOP_BLAS_OP_N ? m : n;

    for (size_t i = 0; i < y_len; ++i) {
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

    if (info.trans == INFINIOP_BLAS_OP_N) {
        for (size_t j = 0; j < n; ++j) {
            const auto temp = alpha_v * x[utils::cast<ptrdiff_t>(j) * incx];
            const auto i_begin = j > ku ? j - ku : 0;
            const auto i_end = std::min(m, j + kl + 1);
            for (size_t i = i_begin; i < i_end; ++i) {
                const auto band_row = ku + i - j;
                const auto A_idx = utils::cast<ptrdiff_t>(band_row) + utils::cast<ptrdiff_t>(j) * lda;
                const auto y_idx = utils::cast<ptrdiff_t>(i) * incy;
                y[y_idx] += temp * A[A_idx];
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            Tdata sum = static_cast<Tdata>(0);
            const auto i_begin = j > ku ? j - ku : 0;
            const auto i_end = std::min(m, j + kl + 1);
            for (size_t i = i_begin; i < i_end; ++i) {
                const auto band_row = ku + i - j;
                const auto A_idx = utils::cast<ptrdiff_t>(band_row) + utils::cast<ptrdiff_t>(j) * lda;
                sum += A[A_idx] * x[utils::cast<ptrdiff_t>(i) * incx];
            }
            const auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += alpha_v * sum;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_GBMV(TDATA)           \
    calculateGbmv(_info,                \
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
        return CALCULATE_GBMV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_GBMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_GBMV

} // namespace op::gbmv::cpu
