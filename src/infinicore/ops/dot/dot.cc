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

    size_t result_size = dsize(dtype);
    void *result = malloc(result_size);

    Dot::execute(result, x, y);

    Shape result_shape = {1};
    auto result_tensor = Tensor::empty(result_shape, dtype, Device::Type::CPU);
    std::memcpy(result_tensor->data(), result, result_size);
    free(result);
    result_tensor = result_tensor->to(x->device());

    return result_tensor->squeeze(0);
}

} // namespace infinicore::op