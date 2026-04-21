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

std::tuple<Tensor, Tensor> rot(Tensor x, Tensor y, void *c, void *s) {
    auto out_x = Tensor::empty(x->shape(), x->dtype(), x->device());
    auto out_y = Tensor::empty(y->shape(), y->dtype(), y->device());
    rot_(x, y, out_x, out_y, c, s);
    return {out_x, out_y};
}

void rot_(Tensor x, Tensor y, Tensor out_x, Tensor out_y, void *c, void *s) {
    Rot::execute(x, y, c, s);
    out_x->copy_from(x);
    out_y->copy_from(y);
}

} // namespace infinicore::op