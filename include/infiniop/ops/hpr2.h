#ifndef __INFINIOP_HPR2_API_H__
#define __INFINIOP_HPR2_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHpr2Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHpr2Descriptor(infiniopHandle_t handle,
                                                                infiniopHpr2Descriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t y_desc,
                                                                infiniopTensorDescriptor_t AP_desc);

__INFINI_C __export infiniStatus_t infiniopGetHpr2WorkspaceSize(infiniopHpr2Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHpr2(infiniopHpr2Descriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *x,
                                                const void *y,
                                                void *AP,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHpr2Descriptor(infiniopHpr2Descriptor_t desc);

#endif
