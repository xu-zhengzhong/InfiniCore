#ifndef __INFINIOP_SPR_API_H__
#define __INFINIOP_SPR_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSprDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSprDescriptor(infiniopHandle_t handle,
                                                               infiniopSprDescriptor_t *desc_ptr,
                                                               infiniopBlasFillMode_t uplo,
                                                               infiniopTensorDescriptor_t alpha_desc,
                                                               infiniopTensorDescriptor_t x_desc,
                                                               infiniopTensorDescriptor_t AP_desc);

__INFINI_C __export infiniStatus_t infiniopGetSprWorkspaceSize(infiniopSprDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSpr(infiniopSprDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               const void *alpha,
                                               const void *x,
                                               void *AP,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySprDescriptor(infiniopSprDescriptor_t desc);

#endif
