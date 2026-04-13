#include "scal_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::scal::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc) {
    
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = x_desc->dtype();

    auto result = ScalInfo::createScalInfo(x_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(), 
        0,
        new Opaque{handle->internal()},
        handle->device, 
        handle->device_id);
        
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *x,
    const void *alpha,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    hpccDataType data_type, alpha_type, execution_type;

    void *alpha_ptr = const_cast<void *>(alpha);

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        data_type = HPCC_R_16F;
        alpha_type = HPCC_R_16F;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_BF16:
        data_type = HPCC_R_16BF;
        alpha_type = HPCC_R_16BF;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_F32:
        data_type = HPCC_R_32F;
        alpha_type = HPCC_R_32F;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_F64:
        data_type = HPCC_R_64F;
        alpha_type = HPCC_R_64F;
        execution_type = HPCC_R_64F;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            
            // Perform in-place scale on Y
            CHECK_MCBLAS(
                hcblasScalEx(
                    handle, _info.getSize(),
                    alpha_ptr, alpha_type,
                    x, data_type, _info.getIncx(),
                    execution_type));
                    
            return INFINI_STATUS_SUCCESS;
        }));
        
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::scal::metax