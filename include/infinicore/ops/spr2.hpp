#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Spr2, const Tensor &, const Tensor &, const Tensor &, Tensor, int);

void spr2_(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor ap, int uplo);

} // namespace infinicore::op
