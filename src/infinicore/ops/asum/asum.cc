#include "infinicore/ops/asum.hpp"
#include "../../utils.hpp"
// #include <iostream>

namespace infinicore::op {

common::OpDispatcher<Asum::schema> &Asum::dispatcher() {
    static common::OpDispatcher<Asum::schema> dispatcher_;
    return dispatcher_;
};

void Asum::execute(void *result, const Tensor &x) {
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(result, x);
}

Tensor asum(const Tensor &x) {
    DataType dtype = x->dtype();
    if (dtype != DataType::F32 && dtype != DataType::F64) {
        throw std::runtime_error("asum only supports F32 and F64 data types.");
    }
    Shape result_shape = {1}; // Asum returns a single index, so the shape is [1]
    auto result_tensor = Tensor::empty(result_shape, dtype, Device::Type::CPU);
    Asum::execute(result_tensor->data(), x);
    result_tensor = result_tensor->to(x->device());

    return result_tensor->squeeze(0);
}

} // namespace infinicore::op
