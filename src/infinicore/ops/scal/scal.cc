#include "infinicore/ops/scal.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Scal::schema> &Scal::dispatcher() {
    static common::OpDispatcher<Scal::schema> dispatcher_;
    return dispatcher_;
};

void Scal::execute(Tensor y, Tensor x, float alpha) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x, alpha);
}

Tensor scal(Tensor x, float alpha) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    scal_(y, x, alpha);
    return y;
}

void scal_(Tensor y, Tensor x, float alpha) {
    Scal::execute(y, x, alpha);
}

} // namespace infinicore::op
