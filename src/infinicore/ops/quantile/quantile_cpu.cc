#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/quantile.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

namespace infinicore::op::quantile_impl::cpu {

// 参考 op::common_cpu:indexToOffset
// 将线性索引转换为内存偏移量（支持非连续张量）
inline size_t _flat_index_to_offset(
    size_t flat_index,
    size_t ndim,
    const Size *shape,
    const Stride *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

// 封装（方便调用）
inline size_t _flat_index_to_offset(
    size_t flat_index,
    const Shape &shape,
    const Strides &strides) {
    return _flat_index_to_offset(
        flat_index,
        shape.size(),
        shape.data(),
        strides.data()
    );
}

template <typename T>
T interpolate_quantile(const std::vector<T> &sorted_data, double q_val, 
                       InterpolationMode mode) {
    size_t n = sorted_data.size();
    if (n == 0) {
        throw std::runtime_error("Cannot compute quantile of empty data");
    }
    
    if (n == 1) {
        return sorted_data[0];
    }
    
    // Map q from [0, 1] to [0, n-1]
    double index = q_val * (n - 1);
    
    if (index <= 0) return sorted_data[0];
    if (index >= n - 1) return sorted_data[n - 1];
    
    size_t lower_idx = utils::cast<size_t>(std::floor(index));
    size_t upper_idx = utils::cast<size_t>(std::ceil(index));
    double fraction = index - lower_idx;
    
    T lower_val = sorted_data[lower_idx];
    T upper_val = sorted_data[upper_idx];

    switch (mode) {
        case InterpolationMode::LINEAR:
            if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                return utils::cast<T>(utils::cast<float>(lower_val) + (fraction * (utils::cast<float>(upper_val) - utils::cast<float>(lower_val))));
            } else {
                return lower_val + fraction * (upper_val - lower_val);
            }
        
        case InterpolationMode::LOWER:
            return lower_val;
        
        case InterpolationMode::HIGHER:
            return upper_val;
        
        case InterpolationMode::NEAREST: {
            double rounded_index = std::round(index);
            return sorted_data[utils::cast<size_t>(rounded_index)];
        }
        
        case InterpolationMode::MIDPOINT:
            if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                return utils::cast<T>((utils::cast<float>(lower_val) + utils::cast<float>(upper_val)) / 2.0f);
            } else {
                return (lower_val + upper_val) / utils::cast<T>(2);
            }
        
        default:
            throw std::runtime_error("Unknown interpolation mode");
    }
}

template <typename T>
void quantile_kernel_flat(const T *input, const float *q_data, T *output, 
                          const Shape &input_shape, const Strides &input_strides,
                          const Shape &output_shape, const Strides &output_strides,
                          size_t total_elements, size_t num_quantiles,
                          InterpolationMode mode) {
    std::vector<T> sorted_data(total_elements);

    // 读取非连续input：直接调用_flat_index_to_offset
    for (size_t k = 0; k < total_elements; ++k) {
        size_t input_offset = _flat_index_to_offset(k, input_shape, input_strides);
        sorted_data[k] = input[input_offset];
    }

    std::sort(sorted_data.begin(), sorted_data.end());
    
    // 写入非连续output：遍历quantile维度，计算偏移量
    for (size_t q_idx = 0; q_idx < num_quantiles; ++q_idx) {
        double q_val = utils::cast<double>(q_data[q_idx]);
        if (q_val < 0.0 || q_val > 1.0) {
            throw std::invalid_argument("q values must be in the range [0, 1]");
        }
        T quantile_val = interpolate_quantile(sorted_data, q_val, mode);
        
        // 计算output偏移量（flat场景下output仅quantile一维）
        size_t output_offset = _flat_index_to_offset(q_idx, output_shape, output_strides);
        output[output_offset] = quantile_val;
    }
}

template <typename T>
void quantile_kernel_dim(const T *input, const float *q_data, T *output,
                         const Shape &input_shape,
                         const Shape &output_shape,
                         const Strides &input_strides,
                         const Strides &output_strides, 
                         size_t num_quantiles, int64_t dim,
                         InterpolationMode mode) {
    auto ndim = input_shape.size();
    size_t dim_size = input_shape[dim];
    
    // 计算需要reduce的次数
    size_t num_reductions = 1;
    std::vector<size_t> reduction_shape; // 除dim外的输入维度
    for (size_t i = 0; i < ndim; ++i) {
        if (utils::cast<int64_t>(i) != dim) {
            num_reductions *= input_shape[i];
            reduction_shape.push_back(input_shape[i]);
        }
    }
    
    // 构造output的reduction维度（去掉quantile的第0维）
    std::vector<size_t> output_reduction_shape(output_shape.begin() + 1, output_shape.end());
    
    // 并行计算每个reduction的quantile
    #pragma omp parallel for
    for (size_t reduction_idx = 0; reduction_idx < num_reductions; ++reduction_idx) {
        // 1. 提取当前reduction的slice数据
        // 步骤1：计算reduction_idx对应的偏移量（不含dim维度）
        std::vector<T> slice_data(dim_size);
        size_t base_offset = 0;
        size_t temp = reduction_idx;
        for (size_t i = ndim; i-- > 0;) {
            if (utils::cast<int64_t>(i) == dim) {
                continue; // 跳过dim维度
            }
            base_offset += (temp % reduction_shape[i]) * input_strides[i];
            temp /= reduction_shape[i];
        }

        // 步骤2：加上dim维度j的偏移量
        for (size_t j = 0; j < dim_size; ++j) {
            size_t input_offset = base_offset + j * input_strides[dim];
            slice_data[j] = input[input_offset];
        }
        
        // 2. 排序slice
        std::sort(slice_data.begin(), slice_data.end());

        // 3. 计算并写入每个quantile值
        for (size_t q_idx = 0; q_idx < num_quantiles; ++q_idx) {
            double q_val = utils::cast<double>(q_data[q_idx]);
            if (q_val < 0.0 || q_val > 1.0) {
                throw std::invalid_argument("q values must be in the range [0, 1]");
            }
            T quantile_val = interpolate_quantile(slice_data, q_val, mode);
            
            // 计算output偏移量：q_idx（第0维） + reduction_idx（剩余维度）
            size_t output_linear_idx = q_idx * output_strides[0] + reduction_idx;
            size_t output_offset = _flat_index_to_offset(output_linear_idx, output_shape, output_strides);
            output[output_offset] = quantile_val;
        }
    }
}

template <typename T>
void compute_quantile(const T *input, const float *q_data, T *output,
                      const Shape &input_shape,
                      const Shape &output_shape,
                      const Strides &input_strides,
                      const Strides &output_strides, 
                      size_t num_quantiles, std::optional<int64_t> dim,
                      bool keepdim, InterpolationMode mode) {
    if (!dim.has_value()) {
        // Flatten场景
        size_t total_elements = 1;
        for (auto s : input_shape) {
            total_elements *= s;
        }
        quantile_kernel_flat(input, q_data, output, input_shape, input_strides, 
                            output_shape, output_strides, total_elements, num_quantiles, mode);
    } else {
        // 按维度reduce场景
        auto ndim = input_shape.size();
        auto dim_normalized = dim.value() < 0 ? dim.value() + ndim : dim.value();
        quantile_kernel_dim(input, q_data, output, input_shape, output_shape, input_strides, output_strides, 
                           num_quantiles, dim_normalized, mode);
    }
}

void calculate(Tensor input, Tensor q, Tensor output, 
              std::optional<int64_t> dim, bool keepdim, 
              InterpolationMode interpolation) {
    auto dtype = input->dtype();
    size_t num_quantiles = q->numel();
    
    switch (dtype) {
        case DataType::F32:
            compute_quantile<float>(
                reinterpret_cast<const float *>(input->data()),
                reinterpret_cast<const float *>(q->data()),
                reinterpret_cast<float *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), num_quantiles, dim, keepdim, interpolation);
            break;
        case DataType::F64:
            compute_quantile<double>(
                reinterpret_cast<const double *>(input->data()),
                reinterpret_cast<const float *>(q->data()),
                reinterpret_cast<double *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), num_quantiles, dim, keepdim, interpolation);
            break;
        case DataType::F16:
        case DataType::BF16:
            throw std::runtime_error("Unsupported f16 and bf16 for quantile.");
            break;
        case DataType::I32:
            compute_quantile<int32_t>(
                reinterpret_cast<const int32_t *>(input->data()),
                reinterpret_cast<const float *>(q->data()),
                reinterpret_cast<int32_t *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), num_quantiles, dim, keepdim, interpolation);
            break;
        case DataType::I64:
            compute_quantile<int64_t>(
                reinterpret_cast<const int64_t *>(input->data()),
                reinterpret_cast<const float *>(q->data()),
                reinterpret_cast<int64_t *>(output->data()),
                input->shape(), output->shape(), input->strides(), output->strides(), num_quantiles, dim, keepdim, interpolation);
            break;
        default:
            throw std::runtime_error("Unsupported data type for quantile.");
    }
}

static bool registered = []() {
    Quantile::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::quantile_impl::cpu