#include "trsm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <vector>

namespace op::trsm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasSideMode_t side,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t B_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = TrsmInfo::createTrsmInfo(side, uplo, trans, diag, alpha_desc, A_desc, B_desc);
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
static Tdata loadTriangular(const TrsmInfo &info, const Tdata *A, size_t row, size_t col) {
    if (row == col && info.diag == INFINIOP_BLAS_DIAG_UNIT) {
        return static_cast<Tdata>(1);
    }

    const bool inside = info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? row <= col : row >= col;
    if (!inside) {
        return static_cast<Tdata>(0);
    }

    return A[utils::cast<ptrdiff_t>(row) * info.A_row_stride
             + utils::cast<ptrdiff_t>(col) * info.A_col_stride];
}

template <typename Tdata>
static Tdata loadOpA(const TrsmInfo &info, const Tdata *A, size_t row, size_t col) {
    if (info.trans == INFINIOP_BLAS_OP_N) {
        return loadTriangular(info, A, row, col);
    }
    return loadTriangular(info, A, col, row);
}

static bool opAUpper(const TrsmInfo &info) {
    if (info.trans == INFINIOP_BLAS_OP_N) {
        return info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER;
    }
    return info.uplo == INFINIOP_BLAS_FILL_MODE_LOWER;
}

template <typename Tdata>
static void solveLeft(
    const TrsmInfo &info,
    const Tdata *A,
    Tdata *X,
    size_t col) {

    const auto m = info.m;
    const bool upper = opAUpper(info);

    if (upper) {
        for (ptrdiff_t i = utils::cast<ptrdiff_t>(m) - 1; i >= 0; --i) {
            auto value = X[static_cast<size_t>(i) + col * m];
            for (size_t k = static_cast<size_t>(i) + 1; k < m; ++k) {
                value -= loadOpA(info, A, static_cast<size_t>(i), k) * X[k + col * m];
            }
            if (info.diag == INFINIOP_BLAS_DIAG_NON_UNIT) {
                value /= loadOpA(info, A, static_cast<size_t>(i), static_cast<size_t>(i));
            }
            X[static_cast<size_t>(i) + col * m] = value;
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            auto value = X[i + col * m];
            for (size_t k = 0; k < i; ++k) {
                value -= loadOpA(info, A, i, k) * X[k + col * m];
            }
            if (info.diag == INFINIOP_BLAS_DIAG_NON_UNIT) {
                value /= loadOpA(info, A, i, i);
            }
            X[i + col * m] = value;
        }
    }
}

template <typename Tdata>
static void solveRight(
    const TrsmInfo &info,
    const Tdata *A,
    Tdata *X,
    size_t row) {

    const auto n = info.n;
    const bool upper = opAUpper(info);

    if (upper) {
        for (size_t j = 0; j < n; ++j) {
            auto value = X[row + j * info.m];
            for (size_t k = 0; k < j; ++k) {
                value -= X[row + k * info.m] * loadOpA(info, A, k, j);
            }
            if (info.diag == INFINIOP_BLAS_DIAG_NON_UNIT) {
                value /= loadOpA(info, A, j, j);
            }
            X[row + j * info.m] = value;
        }
    } else {
        for (ptrdiff_t j = utils::cast<ptrdiff_t>(n) - 1; j >= 0; --j) {
            auto value = X[row + static_cast<size_t>(j) * info.m];
            for (size_t k = static_cast<size_t>(j) + 1; k < n; ++k) {
                value -= X[row + k * info.m] * loadOpA(info, A, k, static_cast<size_t>(j));
            }
            if (info.diag == INFINIOP_BLAS_DIAG_NON_UNIT) {
                value /= loadOpA(info, A, static_cast<size_t>(j), static_cast<size_t>(j));
            }
            X[row + static_cast<size_t>(j) * info.m] = value;
        }
    }
}

template <typename Tdata>
infiniStatus_t calculateTrsm(
    const TrsmInfo &info,
    const Tdata *alpha,
    const Tdata *A,
    Tdata *B) {

    const auto m = info.m;
    const auto n = info.n;
    const auto alpha_v = alpha[0];
    std::vector<Tdata> x(m * n);

    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < m; ++row) {
            x[row + col * m] = alpha_v * B[utils::cast<ptrdiff_t>(row) * info.B_row_stride + utils::cast<ptrdiff_t>(col) * info.B_col_stride];
        }
    }

    if (alpha_v != static_cast<Tdata>(0)) {
        if (info.side == INFINIOP_BLAS_SIDE_LEFT) {
#pragma omp parallel for
            for (ptrdiff_t col = 0; col < utils::cast<ptrdiff_t>(n); ++col) {
                solveLeft(info, A, x.data(), static_cast<size_t>(col));
            }
        } else {
#pragma omp parallel for
            for (ptrdiff_t row = 0; row < utils::cast<ptrdiff_t>(m); ++row) {
                solveRight(info, A, x.data(), static_cast<size_t>(row));
            }
        }
    }

    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < m; ++row) {
            B[utils::cast<ptrdiff_t>(row) * info.B_row_stride
              + utils::cast<ptrdiff_t>(col) * info.B_col_stride]
                = x[row + col * m];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_TRSM(TDATA)           \
    calculateTrsm(_info,                \
                  (const TDATA *)alpha, \
                  (const TDATA *)A,     \
                  (TDATA *)B)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *A,
    void *B,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F32:
        return CALCULATE_TRSM(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_TRSM(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_TRSM

} // namespace op::trsm::cpu
