#include "infinicore/ops/sgn.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Sgn::schema> &Sgn::dispatcher() {
    static common::OpDispatcher<Sgn::schema> dispatcher_;
    return dispatcher_;
};

void Sgn::execute(Tensor input, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output);
}

Tensor sgn(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    sgn_(input, output);
    return output;
}

void sgn_(Tensor input, Tensor output) {
    Sgn::execute(input, output);
}

} // namespace infinicore::op