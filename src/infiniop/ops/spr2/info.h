#ifndef __SPR2_INFO_H__
#define __SPR2_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/spr2.h"

struct Spr2Info {
private:
    Spr2Info() = default;

public:
    size_t n;
    infiniopBlasFillMode_t uplo;
    ptrdiff_t incx;
    ptrdiff_t incy;
    infiniDtype_t data_type;

    static utils::Result<Spr2Info> createSpr2Info(
        infiniopBlasFillMode_t uplo,
        infiniopTensorDescriptor_t alpha_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t AP_desc) {

        CHECK_OR_RETURN(uplo == INFINIOP_BLAS_FILL_MODE_UPPER || uplo == INFINIOP_BLAS_FILL_MODE_LOWER, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(alpha_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(AP_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(alpha_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(AP_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = AP_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(alpha_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(y_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        auto n = x_desc->dim(0);
        CHECK_OR_RETURN(y_desc->dim(0) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(AP_desc->dim(0) == n * (n + 1) / 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(AP_desc->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        auto incx = x_desc->stride(0);
        auto incy = y_desc->stride(0);

        return utils::Result<Spr2Info>(Spr2Info{
            n,
            uplo,
            incx,
            incy,
            data_type});
    }
};

#endif // __SPR2_INFO_H__
