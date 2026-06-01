#include "hbmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <algorithm>
#include <complex>

namespace op::hbmv::cpu {

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
    auto result = HbmvInfo::createHbmvInfo(uplo, k, alpha_desc, A_desc, x_desc, beta_desc, y_desc);
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
static std::complex<Real> loadHbmvValue(
    const HbmvInfo &info,
    const ComplexData<Real> *A,
    size_t row,
    size_t col) {

    const auto k = info.k;
    const auto lda = info.A_col_stride;

    if (row == col) {
        const auto idx = utils::cast<ptrdiff_t>(info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? k : 0)
                       + utils::cast<ptrdiff_t>(col) * lda;
        return {A[idx].real, Real(0)};
    }

    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        if (row < col) {
            const auto idx = utils::cast<ptrdiff_t>(k + row - col) + utils::cast<ptrdiff_t>(col) * lda;
            return {A[idx].real, A[idx].imag};
        }
        const auto idx = utils::cast<ptrdiff_t>(k + col - row) + utils::cast<ptrdiff_t>(row) * lda;
        return {A[idx].real, -A[idx].imag};
    }

    if (row > col) {
        const auto idx = utils::cast<ptrdiff_t>(row - col) + utils::cast<ptrdiff_t>(col) * lda;
        return {A[idx].real, A[idx].imag};
    }
    const auto idx = utils::cast<ptrdiff_t>(col - row) + utils::cast<ptrdiff_t>(row) * lda;
    return {A[idx].real, -A[idx].imag};
}

template <typename Real>
infiniStatus_t calculateHbmvImpl(
    const HbmvInfo &info,
    const ComplexData<Real> *alpha,
    const ComplexData<Real> *A,
    const ComplexData<Real> *x,
    const ComplexData<Real> *beta,
    ComplexData<Real> *y) {

    const auto n = info.n;
    const auto k = info.k;
    const auto incx = info.incx;
    const auto incy = info.incy;
    const auto alpha_v = std::complex<Real>{alpha[0].real, alpha[0].imag};
    const auto beta_v = std::complex<Real>{beta[0].real, beta[0].imag};

    for (size_t row = 0; row < n; ++row) {
        std::complex<Real> mv = {0, 0};

        const auto col_begin = row > k ? row - k : 0;
        const auto col_end = std::min(n, row + k + 1);
        for (size_t col = col_begin; col < col_end; ++col) {
            const auto A_v = loadHbmvValue(info, A, row, col);
            const auto x_v = x[utils::cast<ptrdiff_t>(col) * incx];
            mv += A_v * std::complex<Real>{x_v.real, x_v.imag};
        }

        const auto y_idx = utils::cast<ptrdiff_t>(row) * incy;
        const auto y_v = std::complex<Real>{y[y_idx].real, y[y_idx].imag};
        const auto out = alpha_v * mv + beta_v * y_v;

        y[y_idx] = {out.real(), out.imag()};
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_HBMV(REAL)                            \
    calculateHbmvImpl(_info,                            \
                      (const ComplexData<REAL> *)alpha, \
                      (const ComplexData<REAL> *)A,     \
                      (const ComplexData<REAL> *)x,     \
                      (const ComplexData<REAL> *)beta,  \
                      (ComplexData<REAL> *)y)

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
    case INFINI_DTYPE_C64:
        return CALCULATE_HBMV(float);
    case INFINI_DTYPE_C128:
        return CALCULATE_HBMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_HBMV

} // namespace op::hbmv::cpu
