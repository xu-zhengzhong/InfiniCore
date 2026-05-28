#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Gbmv, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor, int, size_t, size_t);

void gbmv_(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int trans, size_t kl, size_t ku);

} // namespace infinicore::op
