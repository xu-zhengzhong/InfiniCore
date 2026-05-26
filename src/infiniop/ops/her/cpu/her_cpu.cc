#include "her_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <complex>

namespace op::her::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t A_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = HerInfo::createHerInfo(uplo, alpha_desc, x_desc, A_desc);
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
infiniStatus_t calculateHer(
    const HerInfo &info,
    const Real *alpha,
    const ComplexData<Real> *x,
    ComplexData<Real> *A) {

    const auto n = info.n;
    const auto incx = info.incx;
    const auto row_stride = info.A_row_stride;
    const auto col_stride = info.A_col_stride;
    const auto alpha_v = alpha[0];

    if (alpha_v == Real(0)) {
        for (size_t j = 0; j < n; ++j) {
            const auto A_idx = utils::cast<ptrdiff_t>(j) * (row_stride + col_stride);
            A[A_idx].imag = Real(0);
        }
        return INFINI_STATUS_SUCCESS;
    }

    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto xj_v = std::complex<Real>{xj.real, xj.imag};
            const auto temp = alpha_v * std::conj(xj_v);

            for (size_t i = 0; i < j; ++i) {
                const auto xi = x[utils::cast<ptrdiff_t>(i) * incx];
                const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                const auto out = std::complex<Real>{A[A_idx].real, A[A_idx].imag}
                               + std::complex<Real>{xi.real, xi.imag} * temp;
                A[A_idx] = {out.real(), out.imag()};
            }

            const auto A_idx = utils::cast<ptrdiff_t>(j) * (row_stride + col_stride);
            A[A_idx] = {A[A_idx].real + alpha_v * std::norm(xj_v), Real(0)};
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto xj_v = std::complex<Real>{xj.real, xj.imag};
            const auto temp = alpha_v * std::conj(xj_v);

            const auto A_diag_idx = utils::cast<ptrdiff_t>(j) * (row_stride + col_stride);
            A[A_diag_idx] = {A[A_diag_idx].real + alpha_v * std::norm(xj_v), Real(0)};

            for (size_t i = j + 1; i < n; ++i) {
                const auto xi = x[utils::cast<ptrdiff_t>(i) * incx];
                const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                const auto out = std::complex<Real>{A[A_idx].real, A[A_idx].imag}
                               + std::complex<Real>{xi.real, xi.imag} * temp;
                A[A_idx] = {out.real(), out.imag()};
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_HER(REAL)                    \
    calculateHer(_info,                        \
                 (const REAL *)alpha,          \
                 (const ComplexData<REAL> *)x, \
                 (ComplexData<REAL> *)A)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    void *A,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_C64:
        return CALCULATE_HER(float);
    case INFINI_DTYPE_C128:
        return CALCULATE_HER(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_HER

} // namespace op::her::cpu
