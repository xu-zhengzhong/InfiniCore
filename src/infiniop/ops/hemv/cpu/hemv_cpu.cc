#include "hemv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <complex>

namespace op::hemv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = HemvInfo::createHemvInfo(uplo, alpha_desc, A_desc, x_desc, beta_desc, y_desc);
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
static ComplexData<Real> loadHermitianValue(
    const HemvInfo &info,
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
            const auto idx = utils::cast<ptrdiff_t>(row) * row_stride + utils::cast<ptrdiff_t>(col) * col_stride;
            return A[idx];
        }
        const auto idx = utils::cast<ptrdiff_t>(col) * row_stride + utils::cast<ptrdiff_t>(row) * col_stride;
        return {A[idx].real, -A[idx].imag};
    }

    if (row > col) {
        const auto idx = utils::cast<ptrdiff_t>(row) * row_stride + utils::cast<ptrdiff_t>(col) * col_stride;
        return A[idx];
    }
    const auto idx = utils::cast<ptrdiff_t>(col) * row_stride + utils::cast<ptrdiff_t>(row) * col_stride;
    return {A[idx].real, -A[idx].imag};
}

template <typename Real>
infiniStatus_t calculateHemvImpl(
    const HemvInfo &info,
    const ComplexData<Real> *alpha,
    const ComplexData<Real> *A,
    const ComplexData<Real> *x,
    const ComplexData<Real> *beta,
    ComplexData<Real> *y) {

    const auto n = info.n;
    const auto incx = info.incx;
    const auto incy = info.incy;
    const auto alpha_v = std::complex<Real>{alpha[0].real, alpha[0].imag};
    const auto beta_v = std::complex<Real>{beta[0].real, beta[0].imag};

    for (size_t row = 0; row < n; ++row) {
        std::complex<Real> mv = {0, 0};

        for (size_t col = 0; col < n; ++col) {
            const auto A_v = loadHermitianValue(info, A, row, col);
            const auto x_v = x[utils::cast<ptrdiff_t>(col) * incx];
            mv += std::complex<Real>{A_v.real, A_v.imag}
                * std::complex<Real>{x_v.real, x_v.imag};
        }

        const auto y_idx = utils::cast<ptrdiff_t>(row) * incy;
        const auto y_v = std::complex<Real>{y[y_idx].real, y[y_idx].imag};
        const auto out = alpha_v * mv + beta_v * y_v;

        y[y_idx] = {out.real(), out.imag()};
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_HEMV(REAL)                            \
    calculateHemvImpl(_info,                            \
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
        return CALCULATE_HEMV(float);
    case INFINI_DTYPE_C128:
        return CALCULATE_HEMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_HEMV

} // namespace op::hemv::cpu
