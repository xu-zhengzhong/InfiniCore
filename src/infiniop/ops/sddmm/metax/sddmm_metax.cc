#include "sddmm_metax.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::sddmm::metax {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopSpMatDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = c_desc->valuesDesc()->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32);

    auto result = SDDMMInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        c_desc->crowIndicesDesc()->dtype(),
        result.take(),
        c_desc,
        0,
        new Opaque(),
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c_values,
    const void *a,
    const void *b,
    float alpha,
    float beta,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    return launchSDDMM(_info, _c_desc, c_values, a, b, alpha, beta, _index_dtype, stream);
}

} // namespace op::sddmm::metax
