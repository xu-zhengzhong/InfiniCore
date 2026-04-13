#include "infinicore/ops/scal.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Scal::schema> &Scal::dispatcher() {
    static common::OpDispatcher<Scal::schema> dispatcher_;
    return dispatcher_;
};

void Scal::execute(Tensor y, void *alpha) {
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, alpha);
}

Tensor scal(Tensor x, void *alpha) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    scal_(y, x, alpha);
    return y;
}

void scal_(Tensor y, Tensor x, void *alpha) {
    if (alpha == nullptr) {
        throw std::runtime_error("scal requires a non-null alpha pointer.");
    }
    y->copy_from(x);
    Scal::execute(y, alpha);
}

} // namespace infinicore::op
