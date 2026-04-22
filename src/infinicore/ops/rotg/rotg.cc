#include "infinicore/ops/rotg.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rotg::schema> &Rotg::dispatcher() {
    static common::OpDispatcher<Rotg::schema> dispatcher_;
    return dispatcher_;
};

void Rotg::execute(Tensor a, Tensor b, Tensor c, Tensor s) {
    infinicore::context::setDevice(a->device());
    dispatcher().lookup(a->device().getType())(a, b, c, s);
}

void rotg_(Tensor a, Tensor b, Tensor c, Tensor s) {
    Rotg::execute(a, b, c, s);
}

} // namespace infinicore::op