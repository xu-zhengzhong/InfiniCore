#include "blas_amax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::blas_amax::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = BlasAmaxInfo::createBlasAmaxInfo(x_desc);
    CHECK_RESULT(info);

    // Create descriptor
    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateBlasAmax(const BlasAmaxInfo &info, const void *x, int *result) {
    const Tdata *x_ptr = reinterpret_cast<const Tdata *>(x);

    const ptrdiff_t size = info.getSize();

    int max_idx = 0;
    float max_val = 0.0;

    for (ptrdiff_t i = 0; i < size; ++i) {
        size_t idx = i * info.getIncx();
        float current_val;
        if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
            current_val = std::abs(utils::cast<float>(x_ptr[idx]));
        } else {
            current_val = std::abs(x_ptr[idx]);
        }

        if (current_val > max_val) {
            max_val = current_val;
            max_idx = static_cast<int>(i);
        }
    }

    result[0] = max_idx + 1; // Convert to 1-based index

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    int *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F32:
        return calculateBlasAmax<float>(_info, x, result);
    case INFINI_DTYPE_F16:
        return calculateBlasAmax<fp16_t>(_info, x, result);
    case INFINI_DTYPE_BF16:
        return calculateBlasAmax<bf16_t>(_info, x, result);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::blas_amax::cpu