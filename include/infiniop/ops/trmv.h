#ifndef __INFINIOP_TRMV_API_H__
#define __INFINIOP_TRMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopTrmvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTrmvDescriptor(infiniopHandle_t handle,
                                                                infiniopTrmvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopBlasDiagType_t diag,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetTrmvWorkspaceSize(infiniopTrmvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTrmv(infiniopTrmvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *A,
                                                void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTrmvDescriptor(infiniopTrmvDescriptor_t desc);

#endif
