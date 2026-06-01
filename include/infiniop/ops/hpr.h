#ifndef __INFINIOP_HPR_API_H__
#define __INFINIOP_HPR_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHprDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHprDescriptor(infiniopHandle_t handle,
                                                               infiniopHprDescriptor_t *desc_ptr,
                                                               infiniopBlasFillMode_t uplo,
                                                               infiniopTensorDescriptor_t alpha_desc,
                                                               infiniopTensorDescriptor_t x_desc,
                                                               infiniopTensorDescriptor_t AP_desc);

__INFINI_C __export infiniStatus_t infiniopGetHprWorkspaceSize(infiniopHprDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHpr(infiniopHprDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               const void *alpha,
                                               const void *x,
                                               void *AP,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHprDescriptor(infiniopHprDescriptor_t desc);

#endif
