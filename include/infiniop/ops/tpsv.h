#ifndef __INFINIOP_TPSV_API_H__
#define __INFINIOP_TPSV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopTpsvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTpsvDescriptor(infiniopHandle_t handle,
                                                                infiniopTpsvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopBlasDiagType_t diag,
                                                                infiniopTensorDescriptor_t AP_desc,
                                                                infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetTpsvWorkspaceSize(infiniopTpsvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTpsv(infiniopTpsvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *AP,
                                                void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTpsvDescriptor(infiniopTpsvDescriptor_t desc);

#endif
