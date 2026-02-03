#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/rot90.hpp"
#include <cstring>
#include <complex>
#include <omp.h>

namespace infinicore::op::rot90_impl::cpu {

// Helper function to compute linear index from multi-dimensional indices
inline size_t compute_index(const std::vector<size_t> &indices, 
                           const Strides &strides) {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides[i];
    }
    return index;
}

template <typename T>
void rot90_kernel(const T *input, T *output, 
                  const Shape &input_shape,
                  const Shape &output_shape,
                  const Strides &input_strides,
                  const Strides &output_strides, 
                  int k, int64_t dim0, int64_t dim1, size_t total_elements) {
    
    auto ndim = output_shape.size();
    
    int64_t size0 = output_shape[dim0];
    int64_t size1 = output_shape[dim1];
    
#pragma omp parallel for
    for (size_t idx = 0; idx < total_elements; ++idx) {
        // Compute output multi-dimensional index
        std::vector<size_t> out_indices(ndim);
        size_t temp = idx;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            out_indices[i] = temp % output_shape[i];
            temp /= output_shape[i];
        }
        
        // Map output indices to input indices based on rotation
        std::vector<size_t> in_indices = out_indices;
        
        size_t out_i = out_indices[dim0];
        size_t out_j = out_indices[dim1];
        
        size_t in_i, in_j;
        
        switch (k) {
            case 1: // 90 degrees clockwise (from first to second axis)
                // out[i, j] = in[j, size0 - 1 - i]
                in_i = out_j;
                in_j = size0 - 1 - out_i;
                break;
            case 2: // 180 degrees
                // out[i, j] = in[size0-1-i, size1-1-j]
                in_i = size0 - 1 - out_i;
                in_j = size1 - 1 - out_j;
                break;
            case 3: // 270 degrees clockwise (or 90 counter-clockwise)
                // out[i, j] = in[size1 - 1 - j, i]
                in_i = size1 - 1 - out_j;
                in_j = out_i;
                break;
            default:
                in_i = out_i;
                in_j = out_j;
        }
        
        in_indices[dim0] = in_i;
        in_indices[dim1] = in_j;
        
        size_t in_idx = compute_index(in_indices, input_strides);
        output[idx] = input[in_idx];
    }
}

void calculate(Tensor input, Tensor output, int k, const std::vector<int64_t> &dims) {
    auto dtype = input->dtype();
    
    auto ndim = input->ndim();
    auto dim0 = dims[0] < 0 ? dims[0] + ndim : dims[0];
    auto dim1 = dims[1] < 0 ? dims[1] + ndim : dims[1];
    
    size_t total_elements = output->numel();
    
    switch (dtype) {
        case DataType::F32:
            rot90_kernel<float>(
                reinterpret_cast<const float *>(input->data()),
                reinterpret_cast<float *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), k, dim0, dim1, total_elements);
            break;
        case DataType::F64:
            rot90_kernel<double>(
                reinterpret_cast<const double *>(input->data()),
                reinterpret_cast<double *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), k, dim0, dim1, total_elements);
            break;
        case DataType::F16:
        case DataType::BF16:
            rot90_kernel<fp16_t>(
                reinterpret_cast<const fp16_t *>(input->data()),
                reinterpret_cast<fp16_t *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), k, dim0, dim1, total_elements);
            break;
        case DataType::I32:
            rot90_kernel<int32_t>(
                reinterpret_cast<const int32_t *>(input->data()),
                reinterpret_cast<int32_t *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), k, dim0, dim1, total_elements);
            break;
        case DataType::I64:
            rot90_kernel<int64_t>(
                reinterpret_cast<const int64_t *>(input->data()),
                reinterpret_cast<int64_t *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), k, dim0, dim1, total_elements);
            break;
        case DataType::U8:
            rot90_kernel<uint8_t>(
                reinterpret_cast<const uint8_t *>(input->data()),
                reinterpret_cast<uint8_t *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), k, dim0, dim1, total_elements);
            break;
        case DataType::C64:
            rot90_kernel<std::complex<float>>(
                reinterpret_cast<const std::complex<float> *>(input->data()),
                reinterpret_cast<std::complex<float> *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), k, dim0, dim1, total_elements);
            break;
        case DataType::C128:
            rot90_kernel<std::complex<double>>(
                reinterpret_cast<const std::complex<double> *>(input->data()),
                reinterpret_cast<std::complex<double> *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), k, dim0, dim1, total_elements);
            break;
        default:
            throw std::runtime_error("Unsupported data type for rot90.");
    }
}

static bool registered = []() {
    Rot90::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::rot90_impl::cpu