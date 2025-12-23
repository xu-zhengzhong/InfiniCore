#include "infinicore/ops/narrow.hpp"

namespace infinicore:: op {

common::OpDispatcher<Narrow:: schema> &Narrow::dispatcher() {
    static common::OpDispatcher<Narrow::schema> dispatcher_;
    return dispatcher_;
};

void Narrow::execute(Tensor output, Tensor input, int dim, int64_t start, int64_t length) {
    dispatcher().lookup(context:: getDevice().getType())(output, input, dim, start, length);
}

Tensor narrow(Tensor input, int dim, int64_t start, int64_t length) {
    // 计算输出形状
    auto input_shape = input->shape();
    auto output_shape = input_shape;
    
    // 处理负维度
    int ndim = input_shape.size();
    if (dim < 0) {
        dim += ndim;
    }
    
    output_shape[dim] = length;
    
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    narrow_(output, input, dim, start, length);
    return output;
}

void narrow_(Tensor output, Tensor input, int dim, int64_t start, int64_t length) {
    Narrow::execute(output, input, dim, start, length);
}

} // namespace infinicore::op