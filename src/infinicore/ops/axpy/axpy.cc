#include "infinicore/ops/axpy.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Axpy::schema> &Axpy::dispatcher() {
    static common::OpDispatcher<Axpy::schema> dispatcher_;
    return dispatcher_;
};

void Axpy::execute(Tensor y, Tensor x, void *alpha) {
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x, alpha);
}

Tensor axpy(Tensor y, Tensor x, Tensor alpha) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    axpy_(y, x, alpha, out);
    return out;
}

void axpy_(Tensor y, Tensor x, Tensor alpha, Tensor out) {
    auto alpha_cpu = alpha->to(Device::Type::CPU);

    Axpy::execute(y, x, alpha_cpu->data());

    // BLAS level1 axpy overwrites the y tensor, so we need to copy the
    // result back to the out tensor to maintain Infinicore's framework.
    out->copy_from(y);
}

} // namespace infinicore::op
