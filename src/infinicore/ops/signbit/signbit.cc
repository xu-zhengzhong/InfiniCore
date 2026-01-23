#include "infinicore/ops/signbit.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<SignBit::schema> &SignBit::dispatcher() {
    static common::OpDispatcher<SignBit::schema> dispatcher_;
    return dispatcher_;
};

void SignBit::execute(Tensor input, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output);
}

Tensor signbit(Tensor input) {
    auto output = Tensor::empty(input->shape(), DataType::U8, input->device());
    signbit_(input, output);
    return output;
}

void signbit_(Tensor input, Tensor output) {
    SignBit::execute(input, output);
}

} // namespace infinicore::op