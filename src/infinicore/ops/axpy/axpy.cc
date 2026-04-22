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

void axpy_(Tensor y, Tensor x, Tensor alpha) {
    auto alpha_cpu = alpha->to(Device::Type::CPU);

    Axpy::execute(y, x, alpha_cpu->data());
}

} // namespace infinicore::op
