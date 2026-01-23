#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/signbit.hpp"
#include <cmath>
#include <omp.h>

namespace infinicore::op::signbit_impl::cpu {

template <typename T>
void signbit_kernel(const T *input, uint8_t *output, size_t total_elements) {
#pragma omp parallel for
    for (size_t i = 0; i < total_elements; ++i) {
        output[i] = static_cast<uint8_t>(std::signbit(input[i]));
    }
}

template <typename T>
void signbit_16_kernel(const T *input, uint8_t *output, size_t total_elements) {
#pragma omp parallel for
    for (size_t i = 0; i < total_elements; ++i) {
        output[i] = static_cast<uint8_t>(input[i]._v & 0x8000);
    }
}

void calculate(Tensor input, Tensor output) {
    auto dtype = input->dtype();
    size_t total_elements = input->numel();

    if (dtype == DataType::F32 || dtype == DataType::F64) {
        signbit_kernel<float>(
            reinterpret_cast<float *>(input->data()),
            reinterpret_cast<uint8_t *>(output->data()),
            total_elements);
    } else if (dtype == DataType::F16 || dtype == DataType::BF16) {
        signbit_16_kernel<fp16_t>(
            reinterpret_cast<fp16_t *>(input->data()),
            reinterpret_cast<uint8_t *>(output->data()),
            total_elements);
    } else {
        throw std::runtime_error("Unsupported data type for signbit.");
    }
}

static bool registered = []() {
    SignBit::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::signbit_impl::cpu