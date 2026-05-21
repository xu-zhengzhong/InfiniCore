#ifndef __INFINIOP_TRSV_API_H__
#define __INFINIOP_TRSV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopTrsvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTrsvDescriptor(infiniopHandle_t handle,
                                                                infiniopTrsvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopBlasDiagType_t diag,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetTrsvWorkspaceSize(infiniopTrsvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTrsv(infiniopTrsvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *A,
                                                void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTrsvDescriptor(infiniopTrsvDescriptor_t desc);

#endif
