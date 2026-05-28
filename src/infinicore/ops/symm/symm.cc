#include "infinicore/ops/symm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Symm);

Symm::Symm(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int side, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, b, alpha, beta, c);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), a, b, alpha, beta, c, side, uplo);
}

void Symm::execute(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int side, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Symm, a, b, alpha, beta, c, side, uplo);
}

void symm_(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int side, int uplo) {
    Symm::execute(a, b, alpha, beta, c, side, uplo);
}

} // namespace infinicore::op
