#ifndef __TBSV_INFO_H__
#define __TBSV_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/tbsv.h"

struct TbsvInfo {
private:
    TbsvInfo() = default;

public:
    size_t n;
    size_t k;
    infiniopBlasFillMode_t uplo;
    infiniopBlasOperation_t trans;
    infiniopBlasDiagType_t diag;
    ptrdiff_t A_row_stride;
    ptrdiff_t A_col_stride;
    ptrdiff_t incx;
    infiniDtype_t data_type;

    static utils::Result<TbsvInfo> createTbsvInfo(
        infiniopBlasFillMode_t uplo,
        infiniopBlasOperation_t trans,
        infiniopBlasDiagType_t diag,
        size_t k,
        infiniopTensorDescriptor_t A_desc,
        infiniopTensorDescriptor_t x_desc) {

        CHECK_OR_RETURN(uplo == INFINIOP_BLAS_FILL_MODE_UPPER || uplo == INFINIOP_BLAS_FILL_MODE_LOWER, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(trans == INFINIOP_BLAS_OP_N || trans == INFINIOP_BLAS_OP_T, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(diag == INFINIOP_BLAS_DIAG_NON_UNIT || diag == INFINIOP_BLAS_DIAG_UNIT, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(A_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(A_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = A_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(x_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        auto band_rows = A_desc->dim(0);
        auto n = A_desc->dim(1);
        CHECK_OR_RETURN(band_rows >= k + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->dim(0) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto A_row_stride = A_desc->stride(0);
        auto A_col_stride = A_desc->stride(1);
        CHECK_OR_RETURN(A_row_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        auto incx = x_desc->stride(0);

        return utils::Result<TbsvInfo>(TbsvInfo{
            n,
            k,
            uplo,
            trans,
            diag,
            A_row_stride,
            A_col_stride,
            incx,
            data_type});
    }
};

#endif // __TBSV_INFO_H__
