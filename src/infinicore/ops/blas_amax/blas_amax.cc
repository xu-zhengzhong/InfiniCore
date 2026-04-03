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
    int result;
    BlasAmax::execute(&result, x);
    
    auto result_tensor = Tensor::empty(result_shape, DataType::I32, Device::Type::CPU);
    std::memcpy(result_tensor->data(), &result, sizeof(int)); // Copy the result into the tensor
    result_tensor = result_tensor->to(x->device());

    return result_tensor->squeeze(0);
}

// void blas_amax_(Tensor result, const Tensor &x) {
//     BlasAmax::execute(result, x);
// }

} // namespace infinicore::op
