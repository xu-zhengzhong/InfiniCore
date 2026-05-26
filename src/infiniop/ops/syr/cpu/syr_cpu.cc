#include "syr_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::syr::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t A_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SyrInfo::createSyrInfo(uplo, alpha_desc, x_desc, A_desc);
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
infiniStatus_t calculateSyr(
    const SyrInfo &info,
    const Tdata *alpha,
    const Tdata *x,
    Tdata *A) {

    const auto n = info.n;
    const auto incx = info.incx;
    const auto row_stride = info.A_row_stride;
    const auto col_stride = info.A_col_stride;
    const auto alpha_v = alpha[0];

    if (alpha_v == static_cast<Tdata>(0)) {
        return INFINI_STATUS_SUCCESS;
    }

    if (info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER) {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            if (xj != static_cast<Tdata>(0)) {
                const auto temp = alpha_v * xj;
                for (size_t i = 0; i <= j; ++i) {
                    const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                    A[A_idx] += x[utils::cast<ptrdiff_t>(i) * incx] * temp;
                }
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
            if (xj != static_cast<Tdata>(0)) {
                const auto temp = alpha_v * xj;
                for (size_t i = j; i < n; ++i) {
                    const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                    A[A_idx] += x[utils::cast<ptrdiff_t>(i) * incx] * temp;
                }
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SYR(TDATA)           \
    calculateSyr(_info,                \
                 (const TDATA *)alpha, \
                 (const TDATA *)x,     \
                 (TDATA *)A)

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
    case INFINI_DTYPE_F32:
        return CALCULATE_SYR(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SYR(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SYR

} // namespace op::syr::cpu
