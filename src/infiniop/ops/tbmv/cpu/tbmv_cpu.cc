#include "tbmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::tbmv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    size_t k,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = TbmvInfo::createTbmvInfo(uplo, trans, diag, k, A_desc, x_desc);
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
infiniStatus_t calculateTbmv(
    const TbmvInfo &info,
    const Tdata *A,
    Tdata *x) {

    const auto n = info.n;
    const auto k = info.k;
    const auto lda = info.A_col_stride;
    const auto incx = info.incx;
    const bool unit_diag = info.diag == INFINIOP_BLAS_DIAG_UNIT;
    const bool upper = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER;
    const bool transposed = info.trans == INFINIOP_BLAS_OP_T;

    std::vector<Tdata> result(n);

    auto x_at = [&](size_t idx) -> Tdata {
        return x[utils::cast<ptrdiff_t>(idx) * incx];
    };

    auto a_at = [&](size_t row, size_t col) -> Tdata {
        ptrdiff_t band_row;
        if (upper) {
            band_row = utils::cast<ptrdiff_t>(k) + utils::cast<ptrdiff_t>(row) - utils::cast<ptrdiff_t>(col);
        } else {
            band_row = utils::cast<ptrdiff_t>(row) - utils::cast<ptrdiff_t>(col);
        }
        return A[band_row + utils::cast<ptrdiff_t>(col) * lda];
    };

    for (size_t i = 0; i < n; ++i) {
        Tdata sum = static_cast<Tdata>(0);

        if (!transposed) {
            const size_t j_begin = upper ? i : 0;
            const size_t j_end = upper ? std::min(n, i + k + 1) : i + 1;
            const size_t lower_j_begin = i > k ? i - k : 0;
            for (size_t j = upper ? j_begin : lower_j_begin; j < j_end; ++j) {
                const auto xj = x_at(j);
                if (unit_diag && i == j) {
                    sum += xj;
                } else {
                    sum += a_at(i, j) * xj;
                }
            }
        } else {
            const size_t j_begin = upper ? (i > k ? i - k : 0) : i;
            const size_t j_end = upper ? i + 1 : std::min(n, i + k + 1);
            for (size_t j = j_begin; j < j_end; ++j) {
                const auto xj = x_at(j);
                if (unit_diag && i == j) {
                    sum += xj;
                } else {
                    sum += a_at(j, i) * xj;
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

#define CALCULATE_TBMV(TDATA)       \
    calculateTbmv(_info,            \
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
        return CALCULATE_TBMV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_TBMV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_TBMV

} // namespace op::tbmv::cpu
