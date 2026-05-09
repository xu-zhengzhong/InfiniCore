#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/ger.h"

#ifdef ENABLE_CPU_API
#include "cpu/ger_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/ger_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/ger_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateGerDescriptor(
    infiniopHandle_t handle,
    infiniopGerDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t A_desc) {

#define CREATE(CASE, NAMESPACE)                                            \
    case CASE:                                                             \
        return op::ger::NAMESPACE::Descriptor::create(                     \
            handle,                                                        \
            reinterpret_cast<op::ger::NAMESPACE::Descriptor **>(desc_ptr), \
            alpha_desc,                                                    \
            x_desc,                                                        \
            y_desc,                                                        \
            A_desc)

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

__INFINI_C infiniStatus_t infiniopGetGerWorkspaceSize(infiniopGerDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<const op::ger::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopGer(
    infiniopGerDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    const void *y,
    void *A,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                            \
    case CASE:                                                                \
        return reinterpret_cast<const op::ger::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, alpha, x, y, A, stream)

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

__INFINI_C infiniStatus_t infiniopDestroyGerDescriptor(infiniopGerDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        delete reinterpret_cast<const op::ger::NAMESPACE::Descriptor *>(desc); \
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
