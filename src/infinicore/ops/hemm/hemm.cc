#include "infinicore/ops/hemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Hemm);

Hemm::Hemm(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int side, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, b, alpha, beta, c);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), a, b, alpha, beta, c, side, uplo);
}

void Hemm::execute(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int side, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Hemm, a, b, alpha, beta, c, side, uplo);
}

void hemm_(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int side, int uplo) {
    Hemm::execute(a, b, alpha, beta, c, side, uplo);
}

} // namespace infinicore::op
