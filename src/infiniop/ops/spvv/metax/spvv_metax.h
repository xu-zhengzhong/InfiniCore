#ifndef __SPVV_METAX_H__
#define __SPVV_METAX_H__

#include "../spvv.h"

SPVV_DESCRIPTOR(metax);

namespace op::spvv::metax {
infiniStatus_t launchCalculateI32(
    void *y,
    const void *values,
    const void *indices,
    const void *x,
    size_t nnz,
    float alpha,
    float beta,
    void *stream);

infiniStatus_t launchCalculateI64(
    void *y,
    const void *values,
    const void *indices,
    const void *x,
    size_t nnz,
    float alpha,
    float beta,
    void *stream);
} // namespace op::spvv::metax

#endif // __SPVV_METAX_H__
