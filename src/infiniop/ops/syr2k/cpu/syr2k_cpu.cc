#include "syr2k_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::syr2k::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t B_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t C_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = Syr2kInfo::createSyr2kInfo(uplo, trans, alpha_desc, A_desc, B_desc, beta_desc, C_desc);
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
static Tdata loadMatrix(const Syr2kInfo &info,
                        const Tdata *matrix,
                        ptrdiff_t row_stride,
                        ptrdiff_t col_stride,
                        size_t c_index,
                        size_t k_index) {
    ptrdiff_t row;
    ptrdiff_t col;
    if (info.trans == INFINIOP_BLAS_OP_N) {
        row = utils::cast<ptrdiff_t>(c_index);
        col = utils::cast<ptrdiff_t>(k_index);
    } else {
        row = utils::cast<ptrdiff_t>(k_index);
        col = utils::cast<ptrdiff_t>(c_index);
    }
    return matrix[row * row_stride + col * col_stride];
}

template <typename Tdata>
infiniStatus_t calculateSyr2k(
    const Syr2kInfo &info,
    const Tdata *alpha,
    const Tdata *A,
    const Tdata *B,
    const Tdata *beta,
    Tdata *C) {

    const auto n = info.n;
    const auto k = info.k;
    const auto alpha_v = alpha[0];
    const auto beta_v = beta[0];
    const bool upper = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER;

#pragma omp parallel for
    for (ptrdiff_t col = 0; col < utils::cast<ptrdiff_t>(n); ++col) {
        const size_t row_begin = upper ? 0 : static_cast<size_t>(col);
        const size_t row_end = upper ? static_cast<size_t>(col) + 1 : n;

        for (size_t row = row_begin; row < row_end; ++row) {
            const auto c_idx = utils::cast<ptrdiff_t>(row) * info.C_row_stride
                             + col * info.C_col_stride;

            if (alpha_v == static_cast<Tdata>(0)) {
                C[c_idx] = beta_v == static_cast<Tdata>(0)
                             ? static_cast<Tdata>(0)
                             : beta_v * C[c_idx];
                continue;
            }

            Tdata sum_ab = static_cast<Tdata>(0);
            Tdata sum_ba = static_cast<Tdata>(0);
            for (size_t l = 0; l < k; ++l) {
                sum_ab += loadMatrix(info, A, info.A_row_stride, info.A_col_stride, row, l)
                        * loadMatrix(info, B, info.B_row_stride, info.B_col_stride, static_cast<size_t>(col), l);
                sum_ba += loadMatrix(info, B, info.B_row_stride, info.B_col_stride, row, l)
                        * loadMatrix(info, A, info.A_row_stride, info.A_col_stride, static_cast<size_t>(col), l);
            }

            const Tdata update = alpha_v * (sum_ab + sum_ba);
            if (beta_v == static_cast<Tdata>(0)) {
                C[c_idx] = update;
            } else {
                C[c_idx] = update + beta_v * C[c_idx];
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SYR2K(TDATA)           \
    calculateSyr2k(_info,                \
                   (const TDATA *)alpha, \
                   (const TDATA *)A,     \
                   (const TDATA *)B,     \
                   (const TDATA *)beta,  \
                   (TDATA *)C)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *A,
    const void *B,
    const void *beta,
    void *C,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        return CALCULATE_SYR2K(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SYR2K(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SYR2K

} // namespace op::syr2k::cpu
