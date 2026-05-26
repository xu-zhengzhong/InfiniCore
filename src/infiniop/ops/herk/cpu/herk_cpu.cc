#include "herk_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <complex>

namespace op::herk::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t C_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = HerkInfo::createHerkInfo(uplo, trans, alpha_desc, A_desc, beta_desc, C_desc);
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
static std::complex<Real> loadA(const HerkInfo &info, const ComplexData<Real> *A, size_t c_index, size_t k_index) {
    ptrdiff_t row;
    ptrdiff_t col;
    if (info.trans == INFINIOP_BLAS_OP_N) {
        row = utils::cast<ptrdiff_t>(c_index);
        col = utils::cast<ptrdiff_t>(k_index);
    } else {
        row = utils::cast<ptrdiff_t>(k_index);
        col = utils::cast<ptrdiff_t>(c_index);
    }
    const auto value = A[row * info.A_row_stride + col * info.A_col_stride];
    return {value.real, value.imag};
}

template <typename Real>
infiniStatus_t calculateHerk(
    const HerkInfo &info,
    const Real *alpha,
    const ComplexData<Real> *A,
    const Real *beta,
    ComplexData<Real> *C) {

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

            std::complex<Real> sum = {0, 0};
            if (alpha_v != Real(0)) {
                for (size_t l = 0; l < k; ++l) {
                    const auto row_a = loadA(info, A, row, l);
                    const auto col_a = loadA(info, A, static_cast<size_t>(col), l);
                    sum += info.trans == INFINIOP_BLAS_OP_N
                             ? row_a * std::conj(col_a)
                             : std::conj(row_a) * col_a;
                }
            }

            const auto c_v = std::complex<Real>{C[c_idx].real, C[c_idx].imag};
            const auto out = beta_v == Real(0)
                               ? alpha_v * sum
                               : alpha_v * sum + beta_v * c_v;
            C[c_idx] = row == static_cast<size_t>(col)
                         ? ComplexData<Real>{out.real(), Real(0)}
                         : ComplexData<Real>{out.real(), out.imag()};
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_HERK(REAL)                    \
    calculateHerk(_info,                        \
                  (const REAL *)alpha,          \
                  (const ComplexData<REAL> *)A, \
                  (const REAL *)beta,           \
                  (ComplexData<REAL> *)C)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *A,
    const void *beta,
    void *C,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_C64:
        return CALCULATE_HERK(float);
    case INFINI_DTYPE_C128:
        return CALCULATE_HERK(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_HERK

} // namespace op::herk::cpu
