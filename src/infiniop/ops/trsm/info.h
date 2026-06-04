#ifndef __TRSM_INFO_H__
#define __TRSM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/trsm.h"

struct TrsmInfo {
private:
    TrsmInfo() = default;

public:
    size_t m;
    size_t n;
    infiniopBlasSideMode_t side;
    infiniopBlasFillMode_t uplo;
    infiniopBlasOperation_t trans;
    infiniopBlasDiagType_t diag;
    ptrdiff_t A_row_stride;
    ptrdiff_t A_col_stride;
    ptrdiff_t B_row_stride;
    ptrdiff_t B_col_stride;
    infiniDtype_t data_type;

    static utils::Result<TrsmInfo> createTrsmInfo(
        infiniopBlasSideMode_t side,
        infiniopBlasFillMode_t uplo,
        infiniopBlasOperation_t trans,
        infiniopBlasDiagType_t diag,
        infiniopTensorDescriptor_t alpha_desc,
        infiniopTensorDescriptor_t A_desc,
        infiniopTensorDescriptor_t B_desc) {

        CHECK_OR_RETURN(side == INFINIOP_BLAS_SIDE_LEFT || side == INFINIOP_BLAS_SIDE_RIGHT, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(uplo == INFINIOP_BLAS_FILL_MODE_UPPER || uplo == INFINIOP_BLAS_FILL_MODE_LOWER, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(trans == INFINIOP_BLAS_OP_N || trans == INFINIOP_BLAS_OP_T, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(diag == INFINIOP_BLAS_DIAG_NON_UNIT || diag == INFINIOP_BLAS_DIAG_UNIT, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(alpha_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(A_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(B_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(alpha_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(B_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = B_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(alpha_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(A_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        const auto m = B_desc->dim(0);
        const auto n = B_desc->dim(1);
        const auto a_dim = side == INFINIOP_BLAS_SIDE_LEFT ? m : n;
        CHECK_OR_RETURN(A_desc->dim(0) == a_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->dim(1) == a_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);

        const auto A_row_stride = A_desc->stride(0);
        const auto A_col_stride = A_desc->stride(1);
        const auto B_row_stride = B_desc->stride(0);
        const auto B_col_stride = B_desc->stride(1);
        CHECK_OR_RETURN(A_row_stride == 1 || A_col_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(B_row_stride == 1 || B_col_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<TrsmInfo>(TrsmInfo{
            m,
            n,
            side,
            uplo,
            trans,
            diag,
            A_row_stride,
            A_col_stride,
            B_row_stride,
            B_col_stride,
            data_type});
    }
};

#endif // __TRSM_INFO_H__
