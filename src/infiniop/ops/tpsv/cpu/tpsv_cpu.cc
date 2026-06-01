#include "tpsv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::tpsv::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    infiniopTensorDescriptor_t AP_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = TpsvInfo::createTpsvInfo(uplo, trans, diag, AP_desc, x_desc);
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
infiniStatus_t calculateTpsv(
    const TpsvInfo &info,
    const Tdata *AP,
    Tdata *x) {

    const auto n = info.n;
    const auto incx = info.incx;
    const bool unit_diag = info.diag == INFINIOP_BLAS_DIAG_UNIT;
    const bool upper = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER;
    const bool transposed = info.trans == INFINIOP_BLAS_OP_T;

    if (n == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    auto x_ref = [&](size_t idx) -> Tdata & {
        return x[utils::cast<ptrdiff_t>(idx) * incx];
    };
    auto upper_idx = [](size_t row, size_t col) -> ptrdiff_t {
        return utils::cast<ptrdiff_t>(col * (col + 1) / 2 + row);
    };
    auto lower_idx = [n](size_t row, size_t col) -> ptrdiff_t {
        return utils::cast<ptrdiff_t>(col * n - col * (col - 1) / 2 + row - col);
    };

    if (!transposed) {
        if (upper) {
            for (size_t jj = 0; jj < n; ++jj) {
                const size_t j = n - 1 - jj;
                Tdata temp = x_ref(j);
                if (temp != static_cast<Tdata>(0)) {
                    if (!unit_diag) {
                        temp /= AP[upper_idx(j, j)];
                        x_ref(j) = temp;
                    }
                    for (size_t i = j; i-- > 0;) {
                        x_ref(i) -= temp * AP[upper_idx(i, j)];
                    }
                }
            }
        } else {
            for (size_t j = 0; j < n; ++j) {
                Tdata temp = x_ref(j);
                if (temp != static_cast<Tdata>(0)) {
                    if (!unit_diag) {
                        temp /= AP[lower_idx(j, j)];
                        x_ref(j) = temp;
                    }
                    for (size_t i = j + 1; i < n; ++i) {
                        x_ref(i) -= temp * AP[lower_idx(i, j)];
                    }
                }
            }
        }
    } else {
        if (upper) {
            for (size_t j = 0; j < n; ++j) {
                Tdata temp = x_ref(j);
                for (size_t i = 0; i < j; ++i) {
                    temp -= AP[upper_idx(i, j)] * x_ref(i);
                }
                if (!unit_diag) {
                    temp /= AP[upper_idx(j, j)];
                }
                x_ref(j) = temp;
            }
        } else {
            for (size_t jj = 0; jj < n; ++jj) {
                const size_t j = n - 1 - jj;
                Tdata temp = x_ref(j);
                for (size_t i = n - 1; i > j; --i) {
                    temp -= AP[lower_idx(i, j)] * x_ref(i);
                }
                if (!unit_diag) {
                    temp /= AP[lower_idx(j, j)];
                }
                x_ref(j) = temp;
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_TPSV(TDATA)        \
    calculateTpsv(_info,             \
                  (const TDATA *)AP, \
                  (TDATA *)x)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *AP,
    void *x,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        return CALCULATE_TPSV(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_TPSV(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_TPSV

} // namespace op::tpsv::cpu
