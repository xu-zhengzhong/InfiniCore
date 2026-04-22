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

void scal_(Tensor x, void *alpha) {
    if (alpha == nullptr) {
        throw std::runtime_error("scal requires a non-null alpha pointer.");
    }
    Scal::execute(x, alpha);
}

} // namespace infinicore::op
