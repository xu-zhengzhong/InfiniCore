#include "infinicore/ops/log.hpp"

namespace infinicore::op {

common::OpDispatcher<Log::schema> &Log::dispatcher() {
    static common::OpDispatcher<Log::schema> d;
    return d;
}

void Log::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

Tensor log(Tensor input) {
    auto out = Tensor::empty(input->shape(), input->dtype(), input->device());
    log_(out, input);
    return out;
}

void log_(Tensor output, Tensor input) {
    Log::execute(output, input);
}

} // namespace infinicore::op