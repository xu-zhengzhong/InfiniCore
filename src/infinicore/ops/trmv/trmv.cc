#include "infinicore/ops/trmv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Trmv);

Trmv::Trmv(const Tensor &a, Tensor x, int uplo, int trans, int diag) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, x);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), a, x, uplo, trans, diag);
}

void Trmv::execute(const Tensor &a, Tensor x, int uplo, int trans, int diag) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Trmv, a, x, uplo, trans, diag);
}

void trmv_(const Tensor &a, Tensor x, int uplo, int trans, int diag) {
    Trmv::execute(a, x, uplo, trans, diag);
}

} // namespace infinicore::op
