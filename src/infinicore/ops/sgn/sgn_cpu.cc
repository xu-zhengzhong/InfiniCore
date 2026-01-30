#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/sgn.hpp"
#include <cmath>
#include <complex>
#include <omp.h>

namespace infinicore::op::sgn_impl::cpu {

template <typename T>
void sgn_real_kernel(const T *input, T *output, size_t total_elements) {
#pragma omp parallel for
    for (size_t i = 0; i < total_elements; ++i) {
        T val = input[i];
        if (val > 0) {
            output[i] = utils::cast<T>(1);
        } else if (val < 0) {
            output[i] = utils::cast<T>(-1);
        } else {
            output[i] = utils::cast<T>(0);
        }
    }
}

template <typename T>
void sgn_16_kernel(const T *input, T *output, size_t total_elements) {
#pragma omp parallel for
    for (size_t i = 0; i < total_elements; ++i) {
        float val = utils::cast<float>(input[i]);
        T result;
        if (val > 0) {
            result = utils::cast<T>(1.0f);
        } else if (val < 0) {
            result = utils::cast<T>(-1.0f);
        } else {
            result = utils::cast<T>(0.0f);
        }
        output[i] = result;
    }
}

template <typename T>
void sgn_complex_kernel(const std::complex<T> *input, std::complex<T> *output, size_t total_elements) {
#pragma omp parallel for
    for (size_t i = 0; i < total_elements; ++i) {
        std::complex<T> val = input[i];
        T magnitude = std::abs(val);
        if (magnitude == utils::cast<T>(0)) {
            output[i] = std::complex<T>(0, 0);
        } else {
            output[i] = val / magnitude;
        }
    }
}

void calculate(Tensor input, Tensor output) {
    auto dtype = input->dtype();
    size_t total_elements = input->numel();

    if (dtype == DataType::F32) {
        sgn_real_kernel<float>(
            reinterpret_cast<float *>(input->data()),
            reinterpret_cast<float *>(output->data()),
            total_elements);
    } else if (dtype == DataType::F64) {
        sgn_real_kernel<double>(
            reinterpret_cast<double *>(input->data()),
            reinterpret_cast<double *>(output->data()),
            total_elements);
    } else if (dtype == DataType::F16 || dtype == DataType::BF16) {
        sgn_16_kernel<fp16_t>(
            reinterpret_cast<fp16_t *>(input->data()),
            reinterpret_cast<fp16_t *>(output->data()),
            total_elements);
    } else if (dtype == DataType::C64) {
        sgn_complex_kernel<float>(
            reinterpret_cast<std::complex<float> *>(input->data()),
            reinterpret_cast<std::complex<float> *>(output->data()),
            total_elements);
    } else if (dtype == DataType::C128) {
        sgn_complex_kernel<double>(
            reinterpret_cast<std::complex<double> *>(input->data()),
            reinterpret_cast<std::complex<double> *>(output->data()),
            total_elements);
    } else {
        throw std::runtime_error("Unsupported data type for sgn.");
    }
}

static bool registered = []() {
    Sgn::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::sgn_impl::cpu