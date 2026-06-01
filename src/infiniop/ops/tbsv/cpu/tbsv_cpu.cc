#include "tbsv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::tbsv::cpu {

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
    auto result = TbsvInfo::createTbsvInfo(uplo, trans, diag, k, A_desc, x_desc);
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
infiniStatus_t calculateTbsv(
    const TbsvInfo &info,
    const Tdata *A,
    Tdata *x) {

    const auto n = info.n;
    const auto k = info.k;
    const auto lda = info.A_col_stride;
    const auto incx = info.incx;
    const bool non_unit = info.diag == INFINIOP_BLAS_DIAG_NON_UNIT;
    const bool upper = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER;
    const bool transposed = info.trans == INFINIOP_BLAS_OP_T;

    if (n == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    auto a_at = [&](size_t row, size_t col) -> Tdata {
        ptrdiff_t band_row;
        if (upper) {
            band_row = utils::cast<ptrdiff_t>(k) + utils::cast<ptrdiff_t>(row) - utils::cast<ptrdiff_t>(col);
        } else {
            band_row = utils::cast<ptrdiff_t>(row) - utils::cast<ptrdiff_t>(col);
        }
        return A[band_row + utils::cast<ptrdiff_t>(col) * lda];
    };
    auto x_at = [&](size_t idx) -> Tdata {
        return x[utils::cast<ptrdiff_t>(idx) * incx];
    };
    auto store_x = [&](size_t idx, Tdata value) {
        x[utils::cast<ptrdiff_t>(idx) * incx] = value;
    };

    if (!transposed) {
        if (upper) {
            for (size_t jj = 0; jj < n; ++jj) {
                const size_t j = n - 1 - jj;
                Tdata value = x_at(j);
                if (non_unit) {
                    value /= a_at(j, j);
                    store_x(j, value);
                }
                const size_t i_begin = j > k ? j - k : 0;
                for (size_t ii = j; ii > i_begin; --ii) {
                    const size_t i = ii - 1;
                    store_x(i, x_at(i) - value * a_at(i, j));
                }
            }
        } else {
            for (size_t j = 0; j < n; ++j) {
                Tdata value = x_at(j);
                if (non_unit) {
                    value /= a_at(j, j);
                    store_x(j, value);
                }
                const size_t i_end = std::min(n, j + k + 1);
                for (size_t i = j + 1; i < i_end; ++i) {
                    store_x(i, x_at(i) - value * a_at(i, j));
                }
            }
        }
    } else {
        if (upper) {
            for (size_t j = 0; j < n; ++j) {
                Tdata value = x_at(j);
                const size_t i_begin = j > k ? j - k : 0;
                for (size_t i = i_begin; i < j; ++i) {
                    value -= a_at(i, j) * x_at(i);
                }
                if (non_unit) {
                    value /= a_at(j, j);
                }
                store_x(j, value);
            }
        } else {
            for (size_t jj = 0; jj < n; ++jj) {
                const size_t j = n - 1 - jj;
                Tdata value = x_at(j);
                const size_t i_end = std::min(n, j + k + 1);
                for (size_t i = i_end; i > j + 1; --i) {
                    const size_t row = i - 1;
                    value -= a_at(row, j) * x_at(row);
                }
                if (non_unit) {
                    value /= a_at(j, j);
                }
                store_x(j, value);
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_TBSV(TDATA)       \
    calculateTbsv(_info,            \
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
        return CALCULATE_TBSV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_TBSV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_TBSV

} // namespace op::tbsv::cpu
