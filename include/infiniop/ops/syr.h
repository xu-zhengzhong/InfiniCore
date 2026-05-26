#ifndef __INFINIOP_SYR_API_H__
#define __INFINIOP_SYR_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSyrDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSyrDescriptor(infiniopHandle_t handle,
                                                               infiniopSyrDescriptor_t *desc_ptr,
                                                               infiniopBlasFillMode_t uplo,
                                                               infiniopTensorDescriptor_t alpha_desc,
                                                               infiniopTensorDescriptor_t x_desc,
                                                               infiniopTensorDescriptor_t A_desc);

__INFINI_C __export infiniStatus_t infiniopGetSyrWorkspaceSize(infiniopSyrDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSyr(infiniopSyrDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               const void *alpha,
                                               const void *x,
                                               void *A,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySyrDescriptor(infiniopSyrDescriptor_t desc);

#endif
