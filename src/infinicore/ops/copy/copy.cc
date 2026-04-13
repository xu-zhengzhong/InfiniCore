#include "infinicore/ops/copy.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Copy::schema> &Copy::dispatcher() {
    static common::OpDispatcher<Copy::schema> dispatcher_;
    return dispatcher_;
};

void Copy::execute(Tensor x, Tensor y) {
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(x, y);
}

Tensor copy(Tensor x, Tensor y) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    copy_(x, y, out);
    return out;
}

void copy_(Tensor x, Tensor y, Tensor out) {
    Copy::execute(x, y);
    out->copy_from(x);
}

} // namespace infinicore::op