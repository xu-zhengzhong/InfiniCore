#include "infinicore/ops/rotm.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rotm::schema> &Rotm::dispatcher() {
    static common::OpDispatcher<Rotm::schema> dispatcher_;
    return dispatcher_;
};

void Rotm::execute(Tensor x, Tensor y, Tensor param) {
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(x, y, param);
}

std::tuple<Tensor, Tensor> rotm(Tensor x, Tensor y, Tensor param) {
    auto out_x = Tensor::empty(x->shape(), x->dtype(), x->device());
    auto out_y = Tensor::empty(y->shape(), y->dtype(), y->device());
    rotm_(x, y, param, out_x, out_y);
    return {out_x, out_y};
}

void rotm_(Tensor x, Tensor y, Tensor param, Tensor out_x, Tensor out_y) {
    Rotm::execute(x, y, param);
    out_x->copy_from(x);
    out_y->copy_from(y);
}

} // namespace infinicore::op