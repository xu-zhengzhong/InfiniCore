#include "hemm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <complex>

namespace op::hemm::cpu {

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
    auto result = HemmInfo::createHemmInfo(side, uplo, alpha_desc, A_desc, B_desc, beta_desc, C_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Real>
struct ComplexData {
    Real real;
    Real imag;
};

template <typename Real>
static ComplexData<Real> loadHermitian(
    const HemmInfo &info,
    const ComplexData<Real> *A,
    size_t row,
    size_t col) {

    const auto row_stride = info.A_row_stride;
    const auto col_stride = info.A_col_stride;

    if (row == col) {
        const auto idx = utils::cast<ptrdiff_t>(row) * (row_stride + col_stride);
        return {A[idx].real, Real(0)};
    }

    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        if (row < col) {
            const auto idx = utils::cast<ptrdiff_t>(row) * row_stride
                           + utils::cast<ptrdiff_t>(col) * col_stride;
            return A[idx];
        }
        const auto idx = utils::cast<ptrdiff_t>(col) * row_stride
                       + utils::cast<ptrdiff_t>(row) * col_stride;
        return {A[idx].real, -A[idx].imag};
    }

    if (row > col) {
        const auto idx = utils::cast<ptrdiff_t>(row) * row_stride
                       + utils::cast<ptrdiff_t>(col) * col_stride;
        return A[idx];
    }
    const auto idx = utils::cast<ptrdiff_t>(col) * row_stride
                   + utils::cast<ptrdiff_t>(row) * col_stride;
    return {A[idx].real, -A[idx].imag};
}

template <typename Real>
infiniStatus_t calculateHemmImpl(
    const HemmInfo &info,
    const ComplexData<Real> *alpha,
    const ComplexData<Real> *A,
    const ComplexData<Real> *B,
    const ComplexData<Real> *beta,
    ComplexData<Real> *C) {

    const auto m = info.m;
    const auto n = info.n;
    const auto alpha_v = std::complex<Real>{alpha[0].real, alpha[0].imag};
    const auto beta_v = std::complex<Real>{beta[0].real, beta[0].imag};

#pragma omp parallel for
    for (ptrdiff_t index = 0; index < utils::cast<ptrdiff_t>(m * n); ++index) {
        const auto row = static_cast<size_t>(index % utils::cast<ptrdiff_t>(m));
        const auto col = static_cast<size_t>(index / utils::cast<ptrdiff_t>(m));
        const auto c_idx = utils::cast<ptrdiff_t>(row) * info.C_row_stride
                         + utils::cast<ptrdiff_t>(col) * info.C_col_stride;

        std::complex<Real> sum = {0, 0};
        if (alpha_v != std::complex<Real>{0, 0}) {
            if (info.side == INFINIOP_BLAS_SIDE_LEFT) {
                for (size_t k = 0; k < m; ++k) {
                    const auto a_v = loadHermitian(info, A, row, k);
                    const auto b_idx = utils::cast<ptrdiff_t>(k) * info.B_row_stride
                                     + utils::cast<ptrdiff_t>(col) * info.B_col_stride;
                    const auto b_v = B[b_idx];
                    sum += std::complex<Real>{a_v.real, a_v.imag}
                         * std::complex<Real>{b_v.real, b_v.imag};
                }
            } else {
                for (size_t k = 0; k < n; ++k) {
                    const auto b_idx = utils::cast<ptrdiff_t>(row) * info.B_row_stride
                                     + utils::cast<ptrdiff_t>(k) * info.B_col_stride;
                    const auto b_v = B[b_idx];
                    const auto a_v = loadHermitian(info, A, k, col);
                    sum += std::complex<Real>{b_v.real, b_v.imag}
                         * std::complex<Real>{a_v.real, a_v.imag};
                }
            }
        }

        const auto c_v = std::complex<Real>{C[c_idx].real, C[c_idx].imag};
        const auto out = beta_v == std::complex<Real>{0, 0}
                           ? alpha_v * sum
                           : alpha_v * sum + beta_v * c_v;
        C[c_idx] = {out.real(), out.imag()};
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_HEMM(REAL)                            \
    calculateHemmImpl(_info,                            \
                      (const ComplexData<REAL> *)alpha, \
                      (const ComplexData<REAL> *)A,     \
                      (const ComplexData<REAL> *)B,     \
                      (const ComplexData<REAL> *)beta,  \
                      (ComplexData<REAL> *)C)

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
    case INFINI_DTYPE_C64:
        return CALCULATE_HEMM(float);
    case INFINI_DTYPE_C128:
        return CALCULATE_HEMM(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_HEMM

} // namespace op::hemm::cpu
