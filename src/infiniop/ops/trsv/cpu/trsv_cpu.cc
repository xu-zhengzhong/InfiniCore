#include "trsv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::trsv::cpu {

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
    auto result = TrsvInfo::createTrsvInfo(uplo, trans, diag, A_desc, x_desc);
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
infiniStatus_t calculateTrsv(
    const TrsvInfo &info,
    const Tdata *A,
    Tdata *x) {

    const auto n = info.n;
    const auto row_stride = info.A_row_stride;
    const auto col_stride = info.A_col_stride;
    const auto incx = info.incx;
    const bool unit_diag = info.diag == INFINIOP_BLAS_DIAG_UNIT;
    const bool upper = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER;
    const bool transposed = info.trans == INFINIOP_BLAS_OP_T;

    if (n == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    auto a_at = [&](size_t row, size_t col) -> Tdata {
        const auto idx = utils::cast<ptrdiff_t>(row) * row_stride + utils::cast<ptrdiff_t>(col) * col_stride;
        return A[idx];
    };
    auto x_at = [&](size_t idx) -> Tdata {
        return x[utils::cast<ptrdiff_t>(idx) * incx];
    };
    auto store_x = [&](size_t idx, Tdata value) {
        x[utils::cast<ptrdiff_t>(idx) * incx] = value;
    };

    if (!transposed) {
        if (upper) {
            for (size_t ii = 0; ii < n; ++ii) {
                const size_t i = n - 1 - ii;
                Tdata value = x_at(i);
                for (size_t j = i + 1; j < n; ++j) {
                    value -= a_at(i, j) * x_at(j);
                }
                if (!unit_diag) {
                    value /= a_at(i, i);
                }
                store_x(i, value);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                Tdata value = x_at(i);
                for (size_t j = 0; j < i; ++j) {
                    value -= a_at(i, j) * x_at(j);
                }
                if (!unit_diag) {
                    value /= a_at(i, i);
                }
                store_x(i, value);
            }
        }
    } else {
        if (upper) {
            for (size_t i = 0; i < n; ++i) {
                Tdata value = x_at(i);
                for (size_t j = 0; j < i; ++j) {
                    value -= a_at(j, i) * x_at(j);
                }
                if (!unit_diag) {
                    value /= a_at(i, i);
                }
                store_x(i, value);
            }
        } else {
            for (size_t ii = 0; ii < n; ++ii) {
                const size_t i = n - 1 - ii;
                Tdata value = x_at(i);
                for (size_t j = i + 1; j < n; ++j) {
                    value -= a_at(j, i) * x_at(j);
                }
                if (!unit_diag) {
                    value /= a_at(i, i);
                }
                store_x(i, value);
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_TRSV(TDATA)       \
    calculateTrsv(_info,            \
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
        return CALCULATE_TRSV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_TRSV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_TRSV

} // namespace op::trsv::cpu
