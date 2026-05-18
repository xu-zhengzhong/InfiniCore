#ifndef __SPMM_NVIDIA_CUH__
#define __SPMM_NVIDIA_CUH__

#include "../spmm.h"

namespace op::spmm::nvidia {

class Descriptor final : public InfiniopDescriptor {
    SPMM_DESCRIPTOR(nvidia)

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle_,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        infiniopSpMatDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        const void *b,
        float alpha,
        float beta,
        void *stream) const;
};

} // namespace op::spmm::nvidia

#endif // __SPMM_NVIDIA_CUH__
