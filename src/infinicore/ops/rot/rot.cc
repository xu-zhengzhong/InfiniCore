#include "infinicore/ops/rot.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rot::schema> &Rot::dispatcher() {
    static common::OpDispatcher<Rot::schema> dispatcher_;
    return dispatcher_;
};

void Rot::execute(Tensor x, Tensor y, void *c, void *s) {
    if (c == nullptr || s == nullptr) {
        throw std::runtime_error("rot requires non-null c and s pointers.");
    }
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(x, y, c, s);
}

void rot_(Tensor x, Tensor y, void *c, void *s) {
    Rot::execute(x, y, c, s);
}

} // namespace infinicore::op