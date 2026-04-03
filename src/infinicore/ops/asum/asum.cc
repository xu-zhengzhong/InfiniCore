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
    size_t result_size = dsize(dtype);
    void *result = malloc(result_size);

    Asum::execute(result, x);
    
    Shape result_shape = {1}; // Asum returns a single index, so the shape is [1]
    auto result_tensor = Tensor::empty(result_shape, dtype, Device::Type::CPU);
    std::memcpy(result_tensor->data(), result, result_size); // Copy the result into the tensor
    free(result); // Free the temporary result memory
    result_tensor = result_tensor->to(x->device());

    return result_tensor->squeeze(0);
}

// void asum_(Tensor result, const Tensor &x) {
//     Asum::execute(result, x);
// }

} // namespace infinicore::op
