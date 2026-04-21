#include "rotg_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <cmath>

namespace op::rotg::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t s_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = RotgInfo::createRotgInfo(a_desc, b_desc, c_desc, s_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateRotg(void *a, void *b, void *c, void *s) {
    using Tcompute = std::conditional_t<std::is_same_v<Tdata, double>, double, float>;

    Tdata *a_ptr = reinterpret_cast<Tdata *>(a);
    Tdata *b_ptr = reinterpret_cast<Tdata *>(b);
    Tdata *c_ptr = reinterpret_cast<Tdata *>(c);
    Tdata *s_ptr = reinterpret_cast<Tdata *>(s);

    const Tcompute zero = utils::cast<Tcompute>(0.0f);
    const Tcompute one = utils::cast<Tcompute>(1.0f);

    Tcompute a_val = utils::cast<Tcompute>(a_ptr[0]);
    Tcompute b_val = utils::cast<Tcompute>(b_ptr[0]);

    const Tcompute anorm = std::abs(a_val);
    const Tcompute bnorm = std::abs(b_val);

    if (bnorm == zero) {
        c_ptr[0] = utils::cast<Tdata>(one);
        s_ptr[0] = utils::cast<Tdata>(zero);
        b_ptr[0] = utils::cast<Tdata>(zero);
        return INFINI_STATUS_SUCCESS;
    }

    if (anorm == zero) {
        c_ptr[0] = utils::cast<Tdata>(zero);
        s_ptr[0] = utils::cast<Tdata>(one);
        a_ptr[0] = utils::cast<Tdata>(b_val);
        b_ptr[0] = utils::cast<Tdata>(one);
        return INFINI_STATUS_SUCCESS;
    }

    const Tcompute sigma = anorm > bnorm ? std::copysign(one, a_val) : std::copysign(one, b_val);
    const Tcompute r = sigma * std::hypot(a_val, b_val);
    const Tcompute c_val = a_val / r;
    const Tcompute s_val = b_val / r;

    Tcompute z;
    if (anorm > bnorm) {
        z = s_val;
    } else if (c_val != zero) {
        z = one / c_val;
    } else {
        z = one;
    }

    a_ptr[0] = utils::cast<Tdata>(r);
    b_ptr[0] = utils::cast<Tdata>(z);
    c_ptr[0] = utils::cast<Tdata>(c_val);
    s_ptr[0] = utils::cast<Tdata>(s_val);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *a,
    void *b,
    void *c,
    void *s,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return calculateRotg<fp16_t>(a, b, c, s);
    case INFINI_DTYPE_BF16:
        return calculateRotg<bf16_t>(a, b, c, s);
    case INFINI_DTYPE_F32:
        return calculateRotg<float>(a, b, c, s);
    case INFINI_DTYPE_F64:
        return calculateRotg<double>(a, b, c, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::rotg::cpu
