#ifndef __INFINIOP_SBMV_API_H__
#define __INFINIOP_SBMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSbmvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSbmvDescriptor(infiniopHandle_t handle,
                                                                infiniopSbmvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                size_t k,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t y_desc);

__INFINI_C __export infiniStatus_t infiniopGetSbmvWorkspaceSize(infiniopSbmvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSbmv(infiniopSbmvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *x,
                                                const void *beta,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySbmvDescriptor(infiniopSbmvDescriptor_t desc);

#endif
