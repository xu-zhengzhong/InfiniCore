#ifndef __TPMV_INFO_H__
#define __TPMV_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/tpmv.h"

struct TpmvInfo {
private:
    TpmvInfo() = default;

public:
    size_t n;
    infiniopBlasFillMode_t uplo;
    infiniopBlasOperation_t trans;
    infiniopBlasDiagType_t diag;
    ptrdiff_t incx;
    infiniDtype_t data_type;

    static utils::Result<TpmvInfo> createTpmvInfo(
        infiniopBlasFillMode_t uplo,
        infiniopBlasOperation_t trans,
        infiniopBlasDiagType_t diag,
        infiniopTensorDescriptor_t AP_desc,
        infiniopTensorDescriptor_t x_desc) {

        CHECK_OR_RETURN(uplo == INFINIOP_BLAS_FILL_MODE_UPPER || uplo == INFINIOP_BLAS_FILL_MODE_LOWER, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(trans == INFINIOP_BLAS_OP_N || trans == INFINIOP_BLAS_OP_T, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(diag == INFINIOP_BLAS_DIAG_NON_UNIT || diag == INFINIOP_BLAS_DIAG_UNIT, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(AP_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(AP_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = AP_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(x_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        auto n = x_desc->dim(0);
        CHECK_OR_RETURN(AP_desc->dim(0) == n * (n + 1) / 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(AP_desc->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        auto incx = x_desc->stride(0);

        return utils::Result<TpmvInfo>(TpmvInfo{
            n,
            uplo,
            trans,
            diag,
            incx,
            data_type});
    }
};

#endif // __TPMV_INFO_H__
