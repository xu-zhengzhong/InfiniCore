#include "infinicore/ops/blas_amax.hpp"
#include "../../utils.hpp"
// #include <iostream>

namespace infinicore::op {

common::OpDispatcher<BlasAmax::schema> &BlasAmax::dispatcher() {
    static common::OpDispatcher<BlasAmax::schema> dispatcher_;
    return dispatcher_;
};

void BlasAmax::execute(int *result, const Tensor &x) {
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(result, x);
}

Tensor blas_amax(const Tensor &x) {
    Shape result_shape = {1}; // BlasAmax returns a single index, so the shape is [1]

    auto result_tensor = Tensor::empty(result_shape, DataType::I32, Device::Type::CPU);
    BlasAmax::execute(reinterpret_cast<int *>(result_tensor->data()), x);
    result_tensor = result_tensor->to(x->device());

    return result_tensor->squeeze(0);
}

} // namespace infinicore::op
