#include "spmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::spmv::cpu {

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
    auto result = SpmvInfo::createSpmvInfo(uplo, alpha_desc, AP_desc, x_desc, beta_desc, y_desc);
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
infiniStatus_t calculateSpmv(
    const SpmvInfo &info,
    const Tdata *alpha,
    const Tdata *AP,
    const Tdata *x,
    const Tdata *beta,
    Tdata *y) {

    const auto n = info.n;
    const auto incx = info.incx;
    const auto incy = info.incy;
    const auto alpha_v = alpha[0];
    const auto beta_v = beta[0];

    for (size_t i = 0; i < n; ++i) {
        const auto y_idx = utils::cast<ptrdiff_t>(i) * incy;
        if (beta_v == static_cast<Tdata>(0)) {
            y[y_idx] = static_cast<Tdata>(0);
        } else if (beta_v != static_cast<Tdata>(1)) {
            y[y_idx] = beta_v * y[y_idx];
        }
    }

    if (alpha_v == static_cast<Tdata>(0)) {
        return INFINI_STATUS_SUCCESS;
    }

    ptrdiff_t kk = 0;
    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        for (size_t j = 0; j < n; ++j) {
            const auto temp1 = alpha_v * x[utils::cast<ptrdiff_t>(j) * incx];
            Tdata temp2 = static_cast<Tdata>(0);
            auto k = kk;
            for (size_t i = 0; i < j; ++i) {
                const auto AP_v = AP[k];
                const auto y_idx = utils::cast<ptrdiff_t>(i) * incy;
                y[y_idx] += temp1 * AP_v;
                temp2 += AP_v * x[utils::cast<ptrdiff_t>(i) * incx];
                ++k;
            }
            const auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += temp1 * AP[kk + utils::cast<ptrdiff_t>(j)] + alpha_v * temp2;
            kk += utils::cast<ptrdiff_t>(j) + 1;
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto temp1 = alpha_v * x[utils::cast<ptrdiff_t>(j) * incx];
            Tdata temp2 = static_cast<Tdata>(0);
            auto y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += temp1 * AP[kk];

            auto k = kk + 1;
            for (size_t i = j + 1; i < n; ++i) {
                const auto AP_v = AP[k];
                y_idx = utils::cast<ptrdiff_t>(i) * incy;
                y[y_idx] += temp1 * AP_v;
                temp2 += AP_v * x[utils::cast<ptrdiff_t>(i) * incx];
                ++k;
            }
            y_idx = utils::cast<ptrdiff_t>(j) * incy;
            y[y_idx] += alpha_v * temp2;
            kk += utils::cast<ptrdiff_t>(n - j);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SPMV(TDATA)           \
    calculateSpmv(_info,                \
                  (const TDATA *)alpha, \
                  (const TDATA *)AP,    \
                  (const TDATA *)x,     \
                  (const TDATA *)beta,  \
                  (TDATA *)y)

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
    case INFINI_DTYPE_F32:
        return CALCULATE_SPMV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SPMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SPMV

} // namespace op::spmv::cpu
