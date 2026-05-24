#include "symm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::symm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasSideMode_t side,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t B_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t C_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SymmInfo::createSymmInfo(side, uplo, alpha_desc, A_desc, B_desc, beta_desc, C_desc);
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
static Tdata loadSymmetric(const SymmInfo &info, const Tdata *A, size_t row, size_t col) {
    const bool direct = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? row <= col : row >= col;
    const auto a_row = utils::cast<ptrdiff_t>(direct ? row : col);
    const auto a_col = utils::cast<ptrdiff_t>(direct ? col : row);
    return A[a_row * info.A_row_stride + a_col * info.A_col_stride];
}

template <typename Tdata>
infiniStatus_t calculateSymm(
    const SymmInfo &info,
    const Tdata *alpha,
    const Tdata *A,
    const Tdata *B,
    const Tdata *beta,
    Tdata *C) {

    const auto m = info.m;
    const auto n = info.n;
    const auto alpha_v = alpha[0];
    const auto beta_v = beta[0];

#pragma omp parallel for
    for (ptrdiff_t index = 0; index < utils::cast<ptrdiff_t>(m * n); ++index) {
        const auto row = static_cast<size_t>(index % utils::cast<ptrdiff_t>(m));
        const auto col = static_cast<size_t>(index / utils::cast<ptrdiff_t>(m));
        const auto c_idx = utils::cast<ptrdiff_t>(row) * info.C_row_stride
                         + utils::cast<ptrdiff_t>(col) * info.C_col_stride;

        Tdata sum = static_cast<Tdata>(0);
        if (alpha_v != static_cast<Tdata>(0)) {
            if (info.side == INFINIOP_BLAS_SIDE_LEFT) {
                for (size_t k = 0; k < m; ++k) {
                    const auto b_idx = utils::cast<ptrdiff_t>(k) * info.B_row_stride
                                     + utils::cast<ptrdiff_t>(col) * info.B_col_stride;
                    sum += loadSymmetric(info, A, row, k) * B[b_idx];
                }
            } else {
                for (size_t k = 0; k < n; ++k) {
                    const auto b_idx = utils::cast<ptrdiff_t>(row) * info.B_row_stride
                                     + utils::cast<ptrdiff_t>(k) * info.B_col_stride;
                    sum += B[b_idx] * loadSymmetric(info, A, k, col);
                }
            }
        }

        if (beta_v == static_cast<Tdata>(0)) {
            C[c_idx] = alpha_v * sum;
        } else {
            C[c_idx] = alpha_v * sum + beta_v * C[c_idx];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SYMM(TDATA)           \
    calculateSymm(_info,                \
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
        return CALCULATE_SYMM(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SYMM(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SYMM

} // namespace op::symm::cpu
