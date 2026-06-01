#ifndef __INFINIOP_HER2_API_H__
#define __INFINIOP_HER2_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHer2Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHer2Descriptor(infiniopHandle_t handle,
                                                                infiniopHer2Descriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t y_desc,
                                                                infiniopTensorDescriptor_t A_desc);

__INFINI_C __export infiniStatus_t infiniopGetHer2WorkspaceSize(infiniopHer2Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHer2(infiniopHer2Descriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *x,
                                                const void *y,
                                                void *A,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHer2Descriptor(infiniopHer2Descriptor_t desc);

#endif
