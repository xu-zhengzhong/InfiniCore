#include "infinicore/ops/nrm2.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Nrm2::schema> &Nrm2::dispatcher() {
    static common::OpDispatcher<Nrm2::schema> dispatcher_;
    return dispatcher_;
};

void Nrm2::execute(void *result, const Tensor &x) {
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(result, x);
}

Tensor nrm2(const Tensor &x) {
    DataType dtype = x->dtype();
    if (dtype != DataType::F32 && dtype != DataType::F64) {
        throw std::runtime_error("nrm2 only supports F32 and F64 data types.");
    }
    size_t result_size = dsize(dtype);
    void *result = malloc(result_size);

    Nrm2::execute(result, x);
    
    Shape result_shape = {1}; // Nrm2 returns a single index, so the shape is [1]
    auto result_tensor = Tensor::empty(result_shape, dtype, Device::Type::CPU);
    std::memcpy(result_tensor->data(), result, result_size); // Copy the result into the tensor
    free(result); // Free the temporary result memory
    result_tensor = result_tensor->to(x->device());

    return result_tensor->squeeze(0);
}

// void nrm2_(Tensor result, const Tensor &x) {
//     Nrm2::execute(result, x);
// }

} // namespace infinicore::op
