#include "infinicore/ops/log.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Log::schema> &Log::dispatcher() {
    static common::OpDispatcher<Log::schema> dispatcher_;
    return dispatcher_;
};

void Log::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Log implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor log(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    log_(output, input);
    return output;
}

void log_(Tensor output, Tensor input) {
    Log::execute(output, input);
}
} // namespace infinicore::op
