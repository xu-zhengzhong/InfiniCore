#include "trmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::trmv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = TrmvInfo::createTrmvInfo(uplo, trans, diag, A_desc, x_desc);
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
infiniStatus_t calculateTrmv(
    const TrmvInfo &info,
    const Tdata *A,
    Tdata *x) {

    const auto n = info.n;
    const auto row_stride = info.A_row_stride;
    const auto col_stride = info.A_col_stride;
    const auto incx = info.incx;
    const bool unit_diag = info.diag == INFINIOP_BLAS_DIAG_UNIT;

    std::vector<Tdata> result(n);

    for (size_t i = 0; i < n; ++i) {
        Tdata sum = static_cast<Tdata>(0);

        if (info.trans == INFINIOP_BLAS_OP_N) {
            const size_t j_begin = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? i : 0;
            const size_t j_end = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? n : i + 1;
            for (size_t j = j_begin; j < j_end; ++j) {
                const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
                if (unit_diag && i == j) {
                    sum += xj;
                } else {
                    const auto A_idx = utils::cast<ptrdiff_t>(i) * row_stride + utils::cast<ptrdiff_t>(j) * col_stride;
                    sum += A[A_idx] * xj;
                }
            }
        } else {
            const size_t j_begin = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? 0 : i;
            const size_t j_end = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? i + 1 : n;
            for (size_t j = j_begin; j < j_end; ++j) {
                const auto xj = x[utils::cast<ptrdiff_t>(j) * incx];
                if (unit_diag && i == j) {
                    sum += xj;
                } else {
                    const auto A_idx = utils::cast<ptrdiff_t>(j) * row_stride + utils::cast<ptrdiff_t>(i) * col_stride;
                    sum += A[A_idx] * xj;
                }
            }
        }

        result[i] = sum;
    }

    for (size_t i = 0; i < n; ++i) {
        x[utils::cast<ptrdiff_t>(i) * incx] = result[i];
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_TRMV(TDATA)       \
    calculateTrmv(_info,            \
                  (const TDATA *)A, \
                  (TDATA *)x)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *A,
    void *x,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        return CALCULATE_TRMV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_TRMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_TRMV

} // namespace op::trmv::cpu
