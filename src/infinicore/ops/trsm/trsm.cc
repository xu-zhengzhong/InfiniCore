#include "infinicore/ops/trsm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Trsm);

Trsm::Trsm(const Tensor &a, const Tensor &alpha, Tensor b, int side, int uplo, int trans, int diag) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, alpha, b);
    INFINICORE_GRAPH_OP_DISPATCH(b->device().getType(), a, alpha, b, side, uplo, trans, diag);
}

void Trsm::execute(const Tensor &a, const Tensor &alpha, Tensor b, int side, int uplo, int trans, int diag) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Trsm, a, alpha, b, side, uplo, trans, diag);
}

void trsm_(const Tensor &a, const Tensor &alpha, Tensor b, int side, int uplo, int trans, int diag) {
    Trsm::execute(a, alpha, b, side, uplo, trans, diag);
}

} // namespace infinicore::op
