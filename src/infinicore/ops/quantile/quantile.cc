#include "infinicore/ops/quantile.hpp"
#include "../../utils.hpp"
#include <algorithm>

namespace infinicore::op {

common::OpDispatcher<Quantile::schema> &Quantile::dispatcher() {
    static common::OpDispatcher<Quantile::schema> dispatcher_;
    return dispatcher_;
};

void Quantile::execute(Tensor input, Tensor q, Tensor output, 
                      std::optional<int64_t> dim, bool keepdim, 
                      InterpolationMode interpolation) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, q, output, dim, keepdim, interpolation);
}

InterpolationMode parse_interpolation_mode(const std::string &mode) {
    if (mode == "linear") return InterpolationMode::LINEAR;
    if (mode == "lower") return InterpolationMode::LOWER;
    if (mode == "higher") return InterpolationMode::HIGHER;
    if (mode == "nearest") return InterpolationMode::NEAREST;
    if (mode == "midpoint") return InterpolationMode::MIDPOINT;
    throw std::invalid_argument("Invalid interpolation mode: " + mode);
}

Tensor quantile(Tensor input, Tensor q, std::optional<int64_t> dim, 
                bool keepdim, InterpolationMode interpolation) {
    auto q_size = q->numel();
    Shape output_shape;
    
    if (dim.has_value()) {
        // Reduce along specified dimension
        auto ndim = input->ndim();
        auto dim_normalized = dim.value() < 0 ? dim.value() + ndim : dim.value();
        
        if (dim_normalized < 0 || dim_normalized >= static_cast<int64_t>(ndim)) {
            throw std::invalid_argument("dim out of range");
        }
        
        // First dimension is for quantiles
        output_shape.push_back(q_size);
        
        for (size_t i = 0; i < ndim; ++i) {
            if (static_cast<int64_t>(i) == dim_normalized) {
                if (keepdim) {
                    output_shape.push_back(1);
                }
            } else {
                output_shape.push_back(input->shape()[i]);
            }
        }
    } else {
        // Flatten input
        if (keepdim) {
            output_shape.resize(input->shape().size() + 1, 1);
            output_shape[0] = q_size;
        } else {
            output_shape.push_back(q_size);
        }
    }
    
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    quantile_(input, q, output, dim, keepdim, interpolation);
    return output;
}

void quantile_(Tensor input, Tensor q, Tensor output, 
               std::optional<int64_t> dim, bool keepdim, 
               InterpolationMode interpolation) {
    Quantile::execute(input, q, output, dim, keepdim, interpolation);
}

} // namespace infinicore::op