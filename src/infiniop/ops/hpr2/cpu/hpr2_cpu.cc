#include "hpr2_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <complex>

namespace op::hpr2::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t AP_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = Hpr2Info::createHpr2Info(uplo, alpha_desc, x_desc, y_desc, AP_desc);
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
infiniStatus_t calculateHpr2(
    const Hpr2Info &info,
    const ComplexData<Real> *alpha,
    const ComplexData<Real> *x,
    const ComplexData<Real> *y,
    ComplexData<Real> *AP) {

    const auto n = info.n;
    const auto incx = info.incx;
    const auto incy = info.incy;
    const auto alpha_v = std::complex<Real>{alpha[0].real, alpha[0].imag};

    if (alpha_v == std::complex<Real>{0, 0}) {
        return INFINI_STATUS_SUCCESS;
    }

    ptrdiff_t kk = 0;
    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto yj = y[utils::cast<ptrdiff_t>(j) * incy];
            const auto xj_v = std::complex<Real>{xj.real, xj.imag};
            const auto yj_v = std::complex<Real>{yj.real, yj.imag};

            if (xj_v != std::complex<Real>{0, 0} || yj_v != std::complex<Real>{0, 0}) {
                const auto temp1 = alpha_v * std::conj(yj_v);
                const auto temp2 = std::conj(alpha_v * xj_v);
                auto k = kk;
                for (size_t i = 0; i < j; ++i) {
                    const auto xi = x[utils::cast<ptrdiff_t>(i) * incx];
                    const auto yi = y[utils::cast<ptrdiff_t>(i) * incy];
                    const auto out = std::complex<Real>{AP[k].real, AP[k].imag}
                                   + std::complex<Real>{xi.real, xi.imag} * temp1
                                   + std::complex<Real>{yi.real, yi.imag} * temp2;
                    AP[k] = {out.real(), out.imag()};
                    ++k;
                }
                const auto diag_update = xj_v * temp1 + yj_v * temp2;
                AP[kk + utils::cast<ptrdiff_t>(j)] = {
                    AP[kk + utils::cast<ptrdiff_t>(j)].real + Real(std::real(diag_update)),
                    Real(0)};
            } else {
                AP[kk + utils::cast<ptrdiff_t>(j)].imag = Real(0);
            }
            kk += utils::cast<ptrdiff_t>(j) + 1;
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            const auto yj = y[utils::cast<ptrdiff_t>(j) * incy];
            const auto xj_v = std::complex<Real>{xj.real, xj.imag};
            const auto yj_v = std::complex<Real>{yj.real, yj.imag};

            if (xj_v != std::complex<Real>{0, 0} || yj_v != std::complex<Real>{0, 0}) {
                const auto temp1 = alpha_v * std::conj(yj_v);
                const auto temp2 = std::conj(alpha_v * xj_v);
                const auto diag_update = xj_v * temp1 + yj_v * temp2;
                AP[kk] = {
                    AP[kk].real + Real(std::real(diag_update)),
                    Real(0)};

                auto k = kk + 1;
                for (size_t i = j + 1; i < n; ++i) {
                    const auto xi = x[utils::cast<ptrdiff_t>(i) * incx];
                    const auto yi = y[utils::cast<ptrdiff_t>(i) * incy];
                    const auto out = std::complex<Real>{AP[k].real, AP[k].imag}
                                   + std::complex<Real>{xi.real, xi.imag} * temp1
                                   + std::complex<Real>{yi.real, yi.imag} * temp2;
                    AP[k] = {out.real(), out.imag()};
                    ++k;
                }
            } else {
                AP[kk].imag = Real(0);
            }
            kk += utils::cast<ptrdiff_t>(n - j);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_HPR2(REAL)                        \
    calculateHpr2(_info,                            \
                  (const ComplexData<REAL> *)alpha, \
                  (const ComplexData<REAL> *)x,     \
                  (const ComplexData<REAL> *)y,     \
                  (ComplexData<REAL> *)AP)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    const void *y,
    void *AP,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_C64:
        return CALCULATE_HPR2(float);
    case INFINI_DTYPE_C128:
        return CALCULATE_HPR2(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_HPR2

} // namespace op::hpr2::cpu
