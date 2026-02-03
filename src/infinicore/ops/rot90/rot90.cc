#include "infinicore/ops/rot90.hpp"
#include "../../utils.hpp"
#include <iostream>

namespace infinicore::op {

common::OpDispatcher<Rot90::schema> &Rot90::dispatcher() {
    static common::OpDispatcher<Rot90::schema> dispatcher_;
    return dispatcher_;
};

void Rot90::execute(Tensor input, Tensor output, int k, const std::vector<int64_t> &dims) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output, k, dims);
}

Tensor rot90(Tensor input, int k, const std::vector<int64_t> &dims) {
    // // Validate dims
    // if (dims.size() != 2) {
    //     throw std::invalid_argument("dims must contain exactly 2 dimensions");
    // }
    
    auto ndim = input->shape().size();
    auto dim0 = dims[0] < 0 ? dims[0] + ndim : dims[0];
    auto dim1 = dims[1] < 0 ? dims[1] + ndim : dims[1];
    
    // if (dim0 < 0 || dim0 >= static_cast<int64_t>(ndim) || 
    //     dim1 < 0 || dim1 >= static_cast<int64_t>(ndim)) {
    //     throw std::invalid_argument("dims out of range");
    // }
    
    // if (dim0 == dim1) {
    //     throw std::invalid_argument("dims must be different");
    // }
    
    // Normalize k to [0, 3]
    int k_normalized = ((k % 4) + 4) % 4;
    
    // // If k is 0, just return a copy
    // if (k_normalized == 0) {
    //     return input->clone();
    // }
    
    // Calculate output shape
    auto output_shape = input->shape();
    if (k_normalized % 2 == 1) {
        // For 90 and 270 degree rotations, swap the dimensions
        std::swap(output_shape[dim0], output_shape[dim1]);
    }
    
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    rot90_(input, output, k_normalized, dims);
    return output;
}

void rot90_(Tensor input, Tensor output, int k, const std::vector<int64_t> &dims) {
    Rot90::execute(input, output, k, dims);
}

} // namespace infinicore::op