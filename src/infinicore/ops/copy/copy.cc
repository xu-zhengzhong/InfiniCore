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

void copy_(Tensor x, Tensor y) {
    Copy::execute(x, y);
}

} // namespace infinicore::op