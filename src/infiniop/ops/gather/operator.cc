#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gather.h"

// --- backend headers ---
#ifdef ENABLE_CPU_API
#include "cpu/gather_cpu.h"
#endif

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/gather_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/gather_metax.h"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/gather_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. Create descriptor
// =======================================================================
__C infiniStatus_t infiniopCreateGatherDescriptor(
    infiniopHandle_t handle,
    infiniopGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    infiniopTensorDescriptor_t index,
    int dim) {

#define CREATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::gather::NAMESPACE::Descriptor::create(                         \
            handle,                                                               \
            reinterpret_cast<op::gather::NAMESPACE::Descriptor **>(desc_ptr),     \
            output, input, index, dim)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif

#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif

#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif

#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

// =======================================================================
// 2. Get workspace size
// =======================================================================
__C infiniStatus_t infiniopGetGatherWorkspaceSize(
    infiniopGatherDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                   \
    case CASE:                                                                                 \
        *size = reinterpret_cast<op::gather::NAMESPACE::Descriptor *>(desc)->workspaceSize();  \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif

#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif

#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif

#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

// =======================================================================
// 3. Execute (calculate)
// =======================================================================
__C infiniStatus_t infiniopGather(
    infiniopGatherDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *index,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                         \
        return reinterpret_cast<const op::gather::NAMESPACE::Descriptor *>(desc)       \
            ->calculate(workspace, workspace_size, output, input, index, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif

#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif

#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif

#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

// =======================================================================
// 4. Destroy descriptor
// =======================================================================
__C infiniStatus_t infiniopDestroyGatherDescriptor(
    infiniopGatherDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                          \
        delete reinterpret_cast<const op::gather::NAMESPACE::Descriptor *>(desc);       \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif

#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif

#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif

#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}

} // extern "C"