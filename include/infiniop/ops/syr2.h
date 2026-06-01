#ifndef __INFINIOP_SYR2_API_H__
#define __INFINIOP_SYR2_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSyr2Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSyr2Descriptor(infiniopHandle_t handle,
                                                                infiniopSyr2Descriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t y_desc,
                                                                infiniopTensorDescriptor_t A_desc);

__INFINI_C __export infiniStatus_t infiniopGetSyr2WorkspaceSize(infiniopSyr2Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSyr2(infiniopSyr2Descriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *x,
                                                const void *y,
                                                void *A,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySyr2Descriptor(infiniopSyr2Descriptor_t desc);

#endif
