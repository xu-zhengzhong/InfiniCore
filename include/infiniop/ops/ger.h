#ifndef __INFINIOP_GER_API_H__
#define __INFINIOP_GER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGerDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGerDescriptor(infiniopHandle_t handle,
                                                               infiniopGerDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t alpha_desc,
                                                               infiniopTensorDescriptor_t x_desc,
                                                               infiniopTensorDescriptor_t y_desc,
                                                               infiniopTensorDescriptor_t A_desc);

__INFINI_C __export infiniStatus_t infiniopGetGerWorkspaceSize(infiniopGerDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopGer(infiniopGerDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               const void *alpha,
                                               const void *x,
                                               const void *y,
                                               void *a,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGerDescriptor(infiniopGerDescriptor_t desc);

#endif
