#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/hpr2.h"

#ifdef ENABLE_CPU_API
#include "cpu/hpr2_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/hpr2_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/hpr2_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateHpr2Descriptor(
    infiniopHandle_t handle,
    infiniopHpr2Descriptor_t *desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t AP_desc) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::hpr2::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::hpr2::NAMESPACE::Descriptor **>(desc_ptr), \
            uplo,                                                           \
            alpha_desc,                                                     \
            x_desc,                                                         \
            y_desc,                                                         \
            AP_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetHpr2WorkspaceSize(infiniopHpr2Descriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<const op::hpr2::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopHpr2(
    infiniopHpr2Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    const void *y,
    void *AP,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::hpr2::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, alpha, x, y, AP, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyHpr2Descriptor(infiniopHpr2Descriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::hpr2::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
