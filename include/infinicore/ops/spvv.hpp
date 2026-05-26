#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../spvec.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SpVV, Tensor, const SpVec &, const Tensor &, float, float);

Tensor spvv(const SpVec &a, const Tensor &x, float alpha = 1.0f, float beta = 0.0f);
void spvv_(Tensor y, const SpVec &a, const Tensor &x, float alpha, float beta);

} // namespace infinicore::op
