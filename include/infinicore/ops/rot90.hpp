#pragma once
#include "../device.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {

class Rot90 {
public:
    using schema = void (*)(Tensor, Tensor, int, const std::vector<int64_t> &);
    static void execute(Tensor input, Tensor output, int k, const std::vector<int64_t> &dims);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor rot90(Tensor input, int k = 1, const std::vector<int64_t> &dims = {0, 1});
void rot90_(Tensor input, Tensor output, int k = 1, const std::vector<int64_t> &dims = {0, 1});

} // namespace infinicore::op