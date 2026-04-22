#include "infinicore/ops/blas_amin.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<BlasAmin::schema> &BlasAmin::dispatcher() {
    static common::OpDispatcher<BlasAmin::schema> dispatcher_;
    return dispatcher_;
};

void BlasAmin::execute(int *result, const Tensor &x) {
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(result, x);
}

Tensor blas_amin(const Tensor &x) {
    Shape result_shape = {1};
    auto result_tensor = Tensor::empty(result_shape, DataType::I32, Device::Type::CPU);
    BlasAmin::execute(reinterpret_cast<int *>(result_tensor->data()), x);
    result_tensor = result_tensor->to(x->device());

    return result_tensor->squeeze(0);
}

} // namespace infinicore::op