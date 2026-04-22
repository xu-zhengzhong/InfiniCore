#include "infinicore/ops/dot.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Dot::schema> &Dot::dispatcher() {
    static common::OpDispatcher<Dot::schema> dispatcher_;
    return dispatcher_;
};

void Dot::execute(void *result, const Tensor &x, const Tensor &y) {
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(result, x, y);
}

Tensor dot(const Tensor &x, const Tensor &y) {
    if (x->dtype() != y->dtype()) {
        throw std::runtime_error("dot requires x and y to have the same dtype.");
    }

    DataType dtype = x->dtype();
    if (dtype != DataType::F32 && dtype != DataType::F64) {
        throw std::runtime_error("dot only supports F32 and F64 data types.");
    }

    Shape result_shape = {1};
    auto result_tensor = Tensor::empty(result_shape, dtype, Device::Type::CPU);
    Dot::execute(result_tensor->data(), x, y);
    result_tensor = result_tensor->to(x->device());

    return result_tensor->squeeze(0);
}

} // namespace infinicore::op