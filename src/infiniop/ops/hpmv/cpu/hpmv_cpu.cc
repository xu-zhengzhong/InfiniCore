#include "hpmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <complex>

namespace op::hpmv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t AP_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = HpmvInfo::createHpmvInfo(uplo, alpha_desc, AP_desc, x_desc, beta_desc, y_desc);
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
infiniStatus_t calculateHpmvImpl(
    const HpmvInfo &info,
    const ComplexData<Real> *alpha,
    const ComplexData<Real> *AP,
    const ComplexData<Real> *x,
    const ComplexData<Real> *beta,
    ComplexData<Real> *y) {

    const auto n = info.n;
    const auto incx = info.incx;
    const auto incy = info.incy;
    const auto alpha_v = std::complex<Real>{alpha[0].real, alpha[0].imag};
    const auto beta_v = std::complex<Real>{beta[0].real, beta[0].imag};

    if (beta_v != std::complex<Real>{1, 0}) {
        for (size_t i = 0; i < n; ++i) {
            const auto y_idx = utils::cast<ptrdiff_t>(i) * incy;
            if (beta_v == std::complex<Real>{0, 0}) {
                y[y_idx] = {0, 0};
            } else {
                const auto y_v = std::complex<Real>{y[y_idx].real, y[y_idx].imag};
                const auto out = beta_v * y_v;
                y[y_idx] = {out.real(), out.imag()};
            }
        }
    }

    if (alpha_v == std::complex<Real>{0, 0}) {
        return INFINI_STATUS_SUCCESS;
    }

    ptrdiff_t kk = 0;
    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        for (size_t j = 0; j < n; ++j) {
            const auto x_j = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto temp1 = alpha_v * std::complex<Real>{x_j.real, x_j.imag};
            std::complex<Real> temp2 = {0, 0};

            auto k = kk;
            for (size_t i = 0; i < j; ++i) {
                const auto AP_v = std::complex<Real>{AP[k].real, AP[k].imag};
                const auto y_idx = utils::cast<ptrdiff_t>(i) * incy;
                const auto y_i = std::complex<Real>{y[y_idx].real, y[y_idx].imag};
                const auto out_i = y_i + temp1 * AP_v;
                y[y_idx] = {out_i.real(), out_i.imag()};

                const auto x_i = x[utils::cast<ptrdiff_t>(i) * incx];
                temp2 += std::conj(AP_v) * std::complex<Real>{x_i.real, x_i.imag};
                ++k;
            }

            const auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            const auto y_j = std::complex<Real>{y[y_idx].real, y[y_idx].imag};
            const auto diag = std::complex<Real>{AP[kk + utils::cast<ptrdiff_t>(j)].real, Real(0)};
            const auto out_j = y_j + temp1 * diag + alpha_v * temp2;
            y[y_idx] = {out_j.real(), out_j.imag()};

            kk += utils::cast<ptrdiff_t>(j) + 1;
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto x_j = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto temp1 = alpha_v * std::complex<Real>{x_j.real, x_j.imag};
            std::complex<Real> temp2 = {0, 0};

            auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            auto y_j = std::complex<Real>{y[y_idx].real, y[y_idx].imag};
            const auto diag = std::complex<Real>{AP[kk].real, Real(0)};
            auto out_j = y_j + temp1 * diag;
            y[y_idx] = {out_j.real(), out_j.imag()};

            auto k = kk + 1;
            for (size_t i = j + 1; i < n; ++i) {
                const auto AP_v = std::complex<Real>{AP[k].real, AP[k].imag};
                y_idx = utils::cast<ptrdiff_t>(i) * incy;
                const auto y_i = std::complex<Real>{y[y_idx].real, y[y_idx].imag};
                const auto out_i = y_i + temp1 * AP_v;
                y[y_idx] = {out_i.real(), out_i.imag()};

                const auto x_i = x[utils::cast<ptrdiff_t>(i) * incx];
                temp2 += std::conj(AP_v) * std::complex<Real>{x_i.real, x_i.imag};
                ++k;
            }

            y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y_j = std::complex<Real>{y[y_idx].real, y[y_idx].imag};
            out_j = y_j + alpha_v * temp2;
            y[y_idx] = {out_j.real(), out_j.imag()};

            kk += utils::cast<ptrdiff_t>(n - j);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_HPMV(REAL)                            \
    calculateHpmvImpl(_info,                            \
                      (const ComplexData<REAL> *)alpha, \
                      (const ComplexData<REAL> *)AP,    \
                      (const ComplexData<REAL> *)x,     \
                      (const ComplexData<REAL> *)beta,  \
                      (ComplexData<REAL> *)y)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *AP,
    const void *x,
    const void *beta,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_C64:
        return CALCULATE_HPMV(float);
    case INFINI_DTYPE_C128:
        return CALCULATE_HPMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_HPMV

} // namespace op::hpmv::cpu
