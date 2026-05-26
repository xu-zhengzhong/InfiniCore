#include "infinicore/ops/spvv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SpVV);

SpVV::SpVV(Tensor y, const SpVec &a, const Tensor &x, float alpha, float beta) {
    INFINICORE_ASSERT(a);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, a->values(), a->indices(), x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, a, x, alpha, beta);
}

void SpVV::execute(Tensor y, const SpVec &a, const Tensor &x, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SpVV, y, a, x, alpha, beta);
}

Tensor spvv(const SpVec &a, const Tensor &x, float alpha, float beta) {
    INFINICORE_ASSERT(a);
    auto y = Tensor::empty({}, a->dtype(), x->device());
    spvv_(y, a, x, alpha, beta);
    return y;
}

void spvv_(Tensor y, const SpVec &a, const Tensor &x, float alpha, float beta) {
    SpVV::execute(y, a, x, alpha, beta);
}

} // namespace infinicore::op
