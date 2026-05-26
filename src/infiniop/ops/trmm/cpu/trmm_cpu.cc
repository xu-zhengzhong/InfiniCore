#include "trmm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <vector>

namespace op::trmm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasSideMode_t side,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t B_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = TrmmInfo::createTrmmInfo(side, uplo, trans, diag, alpha_desc, A_desc, B_desc);
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
static Tdata loadTriangular(const TrmmInfo &info, const Tdata *A, size_t row, size_t col) {
    if (row == col && info.diag == INFINIOP_BLAS_DIAG_UNIT) {
        return static_cast<Tdata>(1);
    }

    const bool inside = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? row <= col : row >= col;
    if (!inside) {
        return static_cast<Tdata>(0);
    }

    return A[utils::cast<ptrdiff_t>(row) * info.A_row_stride
             + utils::cast<ptrdiff_t>(col) * info.A_col_stride];
}

template <typename Tdata>
static Tdata loadOpA(const TrmmInfo &info, const Tdata *A, size_t row, size_t col) {
    if (info.trans == INFINIOP_BLAS_OP_N) {
        return loadTriangular(info, A, row, col);
    }
    return loadTriangular(info, A, col, row);
}

template <typename Tdata>
infiniStatus_t calculateTrmm(
    const TrmmInfo &info,
    const Tdata *alpha,
    const Tdata *A,
    Tdata *B) {

    const auto m = info.m;
    const auto n = info.n;
    const auto alpha_v = alpha[0];
    std::vector<Tdata> b_copy(m * n);

    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < m; ++row) {
            b_copy[row + col * m] = B[utils::cast<ptrdiff_t>(row) * info.B_row_stride
                                      + utils::cast<ptrdiff_t>(col) * info.B_col_stride];
        }
    }

#pragma omp parallel for
    for (ptrdiff_t index = 0; index < utils::cast<ptrdiff_t>(m * n); ++index) {
        const auto row = static_cast<size_t>(index % utils::cast<ptrdiff_t>(m));
        const auto col = static_cast<size_t>(index / utils::cast<ptrdiff_t>(m));
        const auto b_idx = utils::cast<ptrdiff_t>(row) * info.B_row_stride
                         + utils::cast<ptrdiff_t>(col) * info.B_col_stride;

        Tdata sum = static_cast<Tdata>(0);
        if (alpha_v != static_cast<Tdata>(0)) {
            if (info.side == INFINIOP_BLAS_SIDE_LEFT) {
                for (size_t k = 0; k < m; ++k) {
                    sum += loadOpA(info, A, row, k) * b_copy[k + col * m];
                }
            } else {
                for (size_t k = 0; k < n; ++k) {
                    sum += b_copy[row + k * m] * loadOpA(info, A, k, col);
                }
            }
        }

        B[b_idx] = alpha_v * sum;
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_TRMM(TDATA)           \
    calculateTrmm(_info,                \
                  (const TDATA *)alpha, \
                  (const TDATA *)A,     \
                  (TDATA *)B)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *A,
    void *B,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        return CALCULATE_TRMM(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_TRMM(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_TRMM

} // namespace op::trmm::cpu
