#ifndef __INFINIOP_NARROW_API_H__
#define __INFINIOP_NARROW_API_H__

#include "../operator_descriptor.h"
#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopNarrowDescriptor_t;

__C __export infiniStatus_t infiniopCreateNarrowDescriptor(infiniopHandle_t handle,
                                                           infiniopNarrowDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t output,
                                                           infiniopTensorDescriptor_t input,
                                                           int64_t dim,
                                                           int64_t start,
                                                           int64_t length);

__C __export infiniStatus_t infiniopGetNarrowWorkspaceSize(infiniopNarrowDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopNarrow(infiniopNarrowDescriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *output,
                                           const void *input,
                                           void *stream);

__C __export infiniStatus_t infiniopDestroyNarrowDescriptor(infiniopNarrowDescriptor_t desc);

#endif
