#ifndef __INFINIOP_COPY_API_H__
#define __INFINIOP_COPY_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCopyDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateCopyDescriptor(
    infiniopHandle_t handle,
    infiniopCopyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t y);

__INFINI_C __export infiniStatus_t infiniopGetCopyWorkspaceSize(
    infiniopCopyDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopCopy(
    infiniopCopyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *x,
    const void *y,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyCopyDescriptor(
    infiniopCopyDescriptor_t desc);

#endif // __INFINIOP_COPY_API_H__
