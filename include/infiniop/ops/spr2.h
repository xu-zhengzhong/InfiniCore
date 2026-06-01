#ifndef __INFINIOP_SPR2_API_H__
#define __INFINIOP_SPR2_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSpr2Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSpr2Descriptor(infiniopHandle_t handle,
                                                                infiniopSpr2Descriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t y_desc,
                                                                infiniopTensorDescriptor_t AP_desc);

__INFINI_C __export infiniStatus_t infiniopGetSpr2WorkspaceSize(infiniopSpr2Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSpr2(infiniopSpr2Descriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *x,
                                                const void *y,
                                                void *AP,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySpr2Descriptor(infiniopSpr2Descriptor_t desc);

#endif
