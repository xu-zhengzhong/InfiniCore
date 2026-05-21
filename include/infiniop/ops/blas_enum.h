#ifndef __INFINIOP_BLAS_ENUM_API_H__
#define __INFINIOP_BLAS_ENUM_API_H__

typedef enum {
    INFINIOP_BLAS_OP_N = 0,
    INFINIOP_BLAS_OP_T = 1,
    INFINIOP_BLAS_OP_C = 2
} infiniopBlasOperation_t;

typedef enum {
    INFINIOP_BLAS_FILL_MODE_UPPER = 0,
    INFINIOP_BLAS_FILL_MODE_LOWER = 1
} infiniopBlasFillMode_t;

typedef enum {
    INFINIOP_BLAS_DIAG_NON_UNIT = 0,
    INFINIOP_BLAS_DIAG_UNIT = 1
} infiniopBlasDiagType_t;

typedef enum {
    INFINIOP_BLAS_SIDE_LEFT = 0,
    INFINIOP_BLAS_SIDE_RIGHT = 1
} infiniopBlasSideMode_t;

#endif // __INFINIOP_BLAS_ENUM_API_H__
