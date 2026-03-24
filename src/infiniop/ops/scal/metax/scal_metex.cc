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

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

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
    float alpha,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    // // Standard BLAS scal heavily relies on contiguous memory layout 
    // // or constant 1D striding. For N-dimensional tensors mapped to BLAS, 
    // // we enforce strict contiguity.
    // if (!_info.isYContiguous() || !_info.isXContiguous()) {
    //     return INFINI_STATUS_BAD_TENSOR_STRIDES;
    // }

    hpccDataType data_type, alpha_type, execution_type;
    
    void *alpha_ptr = nullptr;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        data_type = HPCC_R_16F;
        alpha_type = HPCC_R_32F;
        execution_type = HPCC_R_32F;
        alpha_ptr = &alpha;
        break;
    case INFINI_DTYPE_BF16:
        data_type = HPCC_R_16BF;
        alpha_type = HPCC_R_32F;
        execution_type = HPCC_R_32F;
        alpha_ptr = &alpha;
        break;
    case INFINI_DTYPE_F32:
        data_type = HPCC_R_32F;
        alpha_type = HPCC_R_32F;
        execution_type = HPCC_R_32F;
        alpha_ptr = &alpha;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            
            // // If pointers don't match (out-of-place execution), copy X into Y first
            // if (x != y) {
            //     // Safely cast hcblasHandle_t to mcblasHandle_t for the explicit copy functions
            //     mcblasHandle_t m_handle = reinterpret_cast<mcblasHandle_t>(handle);

            //     if (_info.getDtype() == INFINI_DTYPE_F32) {
            //         CHECK_MCBLAS(mcblasScopy(m_handle, n, static_cast<const float*>(x), 1, static_cast<float*>(y), 1));
            //     }
            //     else {
            //         // For 16-bit types, we pack 2 elements into 1 32-bit float for mcblasScopy.
            //         // Note: If 'n' is oddly sized, this safely copies the even pairs.
            //         // In real-world LLM workloads, embedding/hidden dims are virtually always even.
            //         CHECK_MCBLAS(mcblasScopy(m_handle, n / 2, static_cast<const float*>(x), 1, static_cast<float*>(y), 1));
            //     }
            // }
            
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