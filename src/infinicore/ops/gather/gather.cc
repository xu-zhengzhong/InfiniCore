#include "infinicore/ops/gather.hpp"

namespace infinicore::op {

common::OpDispatcher<Gather::schema> &Gather::dispatcher() {
    static common::OpDispatcher<Gather::schema> dispatcher_;
    return dispatcher_;
}

void Gather::execute(Tensor output, Tensor input, Tensor index, int dim) {
    dispatcher().lookup(context::getDevice().getType())(output, input, index, dim);
}

Tensor gather(Tensor input, Tensor index, int dim) {
    auto output = Tensor::empty(index->shape(), input->dtype(), input->device());
    gather_(output, input, index, dim);
    return output;
}

void gather_(Tensor output, Tensor input, Tensor index, int dim) {
    Gather::execute(output, input, index, dim);
}

} // namespace infinicore::op