#include "infinicore/ops/rotmg.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rotmg::schema> &Rotmg::dispatcher() {
    static common::OpDispatcher<Rotmg::schema> dispatcher_;
    return dispatcher_;
};

void Rotmg::execute(Tensor d1, Tensor d2, Tensor x1, Tensor y1, Tensor param) {
    infinicore::context::setDevice(d1->device());
    dispatcher().lookup(d1->device().getType())(d1, d2, x1, y1, param);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> rotmg(Tensor d1, Tensor d2, Tensor x1, Tensor y1) {
    auto out_d1 = Tensor::empty(d1->shape(), d1->dtype(), d1->device());
    auto out_d2 = Tensor::empty(d2->shape(), d2->dtype(), d2->device());
    auto out_x1 = Tensor::empty(x1->shape(), x1->dtype(), x1->device());
    auto out_param = Tensor::zeros({5}, d1->dtype(), d1->device());
    rotmg_(d1, d2, x1, y1, out_d1, out_d2, out_x1, out_param);
    return {out_d1, out_d2, out_x1, out_param};
}

void rotmg_(
    Tensor d1,
    Tensor d2,
    Tensor x1,
    Tensor y1,
    Tensor out_d1,
    Tensor out_d2,
    Tensor out_x1,
    Tensor out_param) {
    out_d1->copy_from(d1);
    out_d2->copy_from(d2);
    out_x1->copy_from(x1);
    out_param->copy_from(Tensor::zeros(out_param->shape(), out_param->dtype(), out_param->device()));
    Rotmg::execute(out_d1, out_d2, out_x1, y1, out_param);
}

} // namespace infinicore::op