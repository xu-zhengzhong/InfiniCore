#include "spr_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::spr::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t AP_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SprInfo::createSprInfo(uplo, alpha_desc, x_desc, AP_desc);
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
infiniStatus_t calculateSpr(
    const SprInfo &info,
    const Tdata *alpha,
    const Tdata *x,
    Tdata *AP) {

    const auto n = info.n;
    const auto incx = info.incx;
    const auto alpha_v = alpha[0];

    if (alpha_v == static_cast<Tdata>(0)) {
        return INFINI_STATUS_SUCCESS;
    }

    ptrdiff_t kk = 0;
    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            if (xj != static_cast<Tdata>(0)) {
                const auto temp = alpha_v * xj;
                auto k = kk;
                for (size_t i = 0; i <= j; ++i) {
                    AP[k] += x[utils::cast<ptrdiff_t>(i) * incx] * temp;
                    ++k;
                }
            }
            kk += utils::cast<ptrdiff_t>(j) + 1;
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            if (xj != static_cast<Tdata>(0)) {
                const auto temp = alpha_v * xj;
                auto k = kk;
                for (size_t i = j; i < n; ++i) {
                    AP[k] += x[utils::cast<ptrdiff_t>(i) * incx] * temp;
                    ++k;
                }
            }
            kk += utils::cast<ptrdiff_t>(n - j);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SPR(TDATA)           \
    calculateSpr(_info,                \
                 (const TDATA *)alpha, \
                 (const TDATA *)x,     \
                 (TDATA *)AP)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    void *AP,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        return CALCULATE_SPR(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SPR(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SPR

} // namespace op::spr::cpu
