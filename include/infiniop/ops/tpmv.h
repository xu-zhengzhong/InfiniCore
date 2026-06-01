#ifndef __INFINIOP_TPMV_API_H__
#define __INFINIOP_TPMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopTpmvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTpmvDescriptor(infiniopHandle_t handle,
                                                                infiniopTpmvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopBlasDiagType_t diag,
                                                                infiniopTensorDescriptor_t AP_desc,
                                                                infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetTpmvWorkspaceSize(infiniopTpmvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTpmv(infiniopTpmvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *AP,
                                                void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTpmvDescriptor(infiniopTpmvDescriptor_t desc);

#endif
