#include "infinicore/ops/trmm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Trmm);

Trmm::Trmm(const Tensor &a, const Tensor &alpha, Tensor b, int side, int uplo, int trans, int diag) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, alpha, b);
    INFINICORE_GRAPH_OP_DISPATCH(b->device().getType(), a, alpha, b, side, uplo, trans, diag);
}

void Trmm::execute(const Tensor &a, const Tensor &alpha, Tensor b, int side, int uplo, int trans, int diag) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Trmm, a, alpha, b, side, uplo, trans, diag);
}

void trmm_(const Tensor &a, const Tensor &alpha, Tensor b, int side, int uplo, int trans, int diag) {
    Trmm::execute(a, alpha, b, side, uplo, trans, diag);
}

} // namespace infinicore::op
