#ifndef __INFINIOP_SPMV_API_H__
#define __INFINIOP_SPMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSpmvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSpmvDescriptor(infiniopHandle_t handle,
                                                                infiniopSpmvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t AP_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t y_desc);

__INFINI_C __export infiniStatus_t infiniopGetSpmvWorkspaceSize(infiniopSpmvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSpmv(infiniopSpmvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *AP,
                                                const void *x,
                                                const void *beta,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySpmvDescriptor(infiniopSpmvDescriptor_t desc);

#endif
