#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Hbmv, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor, int, size_t);

void hbmv_(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo, size_t k);

} // namespace infinicore::op
