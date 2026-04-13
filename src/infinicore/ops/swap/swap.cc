#include "infinicore/ops/swap.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Swap::schema> &Swap::dispatcher() {
    static common::OpDispatcher<Swap::schema> dispatcher_;
    return dispatcher_;
};

void Swap::execute(Tensor x, Tensor y) {
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(x, y);
}

std::tuple<Tensor, Tensor> swap(Tensor x, Tensor y) {
    auto out_x = Tensor::empty(x->shape(), x->dtype(), x->device());
    auto out_y = Tensor::empty(y->shape(), y->dtype(), y->device());
    swap_(x, y, out_x, out_y);
    return {out_x, out_y};
}

void swap_(Tensor x, Tensor y, Tensor out_x, Tensor out_y) {
    Swap::execute(x, y);
    out_x->copy_from(x);
    out_y->copy_from(y);
}

} // namespace infinicore::op