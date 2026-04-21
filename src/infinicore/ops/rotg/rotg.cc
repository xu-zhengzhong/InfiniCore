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

std::tuple<Tensor, Tensor, Tensor, Tensor> rotg(Tensor a, Tensor b) {
    auto out_a = Tensor::empty(a->shape(), a->dtype(), a->device());
    auto out_b = Tensor::empty(b->shape(), b->dtype(), b->device());
    auto out_c = Tensor::zeros(a->shape(), a->dtype(), a->device());
    auto out_s = Tensor::zeros(b->shape(), b->dtype(), b->device());
    rotg_(a, b, out_a, out_b, out_c, out_s);
    return {out_a, out_b, out_c, out_s};
}

void rotg_(Tensor a, Tensor b, Tensor out_a, Tensor out_b, Tensor out_c, Tensor out_s) {
    out_a->copy_from(a);
    out_b->copy_from(b);
    out_c->copy_from(Tensor::zeros(out_c->shape(), out_c->dtype(), out_c->device()));
    out_s->copy_from(Tensor::zeros(out_s->shape(), out_s->dtype(), out_s->device()));
    Rotg::execute(out_a, out_b, out_c, out_s);
}

} // namespace infinicore::op